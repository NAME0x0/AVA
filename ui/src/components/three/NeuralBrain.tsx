"use client";

import { useRef, useMemo, useEffect, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Sphere, Line, Float } from "@react-three/drei";
import * as THREE from "three";

// Check WebGL availability
function isWebGLAvailable(): boolean {
  try {
    const canvas = document.createElement("canvas");
    return !!(
      window.WebGLRenderingContext &&
      (canvas.getContext("webgl") || canvas.getContext("experimental-webgl"))
    );
  } catch {
    return false;
  }
}

// Fallback component when WebGL is unavailable
function WebGLFallback({ className = "" }: { className?: string }) {
  return (
    <div className={`flex items-center justify-center ${className}`}>
      <div className="text-center p-4 bg-neural-elevated/50 rounded-lg border border-neural-hover">
        <div className="text-2xl mb-2">🧠</div>
        <p className="text-text-muted text-sm">
          3D visualization unavailable
        </p>
        <p className="text-text-muted text-xs mt-1">
          WebGL not supported in this browser
        </p>
      </div>
    </div>
  );
}

interface NeuronProps {
  position: [number, number, number];
  color: string;
  scale?: number;
  pulseSpeed?: number;
  isActive?: boolean;
}

function Neuron({ position, color, scale = 1, pulseSpeed = 1, isActive = false }: NeuronProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const glowRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (meshRef.current) {
      // Pulse effect
      const pulse = Math.sin(state.clock.elapsedTime * pulseSpeed) * 0.1 + 1;
      meshRef.current.scale.setScalar(scale * pulse);

      // Active neurons glow brighter
      if (isActive && glowRef.current) {
        const glow = Math.sin(state.clock.elapsedTime * 3) * 0.5 + 1.5;
        glowRef.current.scale.setScalar(scale * glow * 1.5);
      }
    }
  });

  return (
    <group position={position}>
      {/* Glow effect */}
      <mesh ref={glowRef}>
        <sphereGeometry args={[0.15, 16, 16]} />
        <meshBasicMaterial color={color} transparent opacity={isActive ? 0.3 : 0.1} />
      </mesh>
      {/* Core neuron */}
      <mesh ref={meshRef}>
        <sphereGeometry args={[0.08, 32, 32]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={isActive ? 0.8 : 0.3}
          metalness={0.5}
          roughness={0.2}
        />
      </mesh>
    </group>
  );
}

interface SynapseProps {
  start: [number, number, number];
  end: [number, number, number];
  color: string;
  activity?: number;
}

function Synapse({ start, end, color, activity = 0 }: SynapseProps) {
  const lineRef = useRef<THREE.Line>(null);

  useFrame((state) => {
    if (lineRef.current && lineRef.current.material) {
      const mat = lineRef.current.material as THREE.LineBasicMaterial;
      mat.opacity = 0.2 + activity * 0.6 + Math.sin(state.clock.elapsedTime * 2) * 0.1;
    }
  });

  return (
    <Line
      ref={lineRef}
      points={[start, end]}
      color={color}
      lineWidth={1 + activity * 2}
      transparent
      opacity={0.3}
    />
  );
}

interface NeuralNetworkProps {
  activeComponent: "medulla" | "cortex" | "bridge" | "idle";
  cognitiveState: string;
  entropy: number;
  activity: number;
}

function NeuralNetwork({ activeComponent, cognitiveState, entropy, activity }: NeuralNetworkProps) {
  const groupRef = useRef<THREE.Group>(null);

  // Generate neuron positions
  const neurons = useMemo(() => {
    const positions: { pos: [number, number, number]; region: string }[] = [];

    // Medulla (brainstem) - bottom cluster
    for (let i = 0; i < 12; i++) {
      const theta = (i / 12) * Math.PI * 2;
      const r = 0.5 + Math.random() * 0.3;
      positions.push({
        pos: [Math.cos(theta) * r, -1.5 + Math.random() * 0.5, Math.sin(theta) * r],
        region: "medulla",
      });
    }

    // Bridge (connection) - middle
    for (let i = 0; i < 8; i++) {
      const y = -0.8 + (i / 8) * 1.2;
      const theta = (i / 8) * Math.PI * 2 + Math.random() * 0.5;
      const r = 0.3;
      positions.push({
        pos: [Math.cos(theta) * r, y, Math.sin(theta) * r],
        region: "bridge",
      });
    }

    // Cortex (neocortex) - top hemisphere
    for (let i = 0; i < 30; i++) {
      const phi = Math.acos(1 - 2 * (i / 30));
      const theta = Math.sqrt(30 * Math.PI) * phi;
      const r = 1.2;
      positions.push({
        pos: [
          r * Math.sin(phi) * Math.cos(theta),
          0.8 + r * Math.cos(phi) * 0.6,
          r * Math.sin(phi) * Math.sin(theta),
        ],
        region: "cortex",
      });
    }

    return positions;
  }, []);

  // Generate synapses
  const synapses = useMemo(() => {
    const connections: { start: [number, number, number]; end: [number, number, number]; region: string }[] = [];

    neurons.forEach((n1, i) => {
      neurons.forEach((n2, j) => {
        if (i < j) {
          const dist = Math.sqrt(
            Math.pow(n1.pos[0] - n2.pos[0], 2) +
            Math.pow(n1.pos[1] - n2.pos[1], 2) +
            Math.pow(n1.pos[2] - n2.pos[2], 2)
          );
          // Connect nearby neurons
          if (dist < 0.8 && Math.random() > 0.5) {
            connections.push({
              start: n1.pos,
              end: n2.pos,
              region: n1.region === n2.region ? n1.region : "bridge",
            });
          }
        }
      });
    });

    return connections;
  }, [neurons]);

  // Rotate slowly
  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = state.clock.elapsedTime * 0.1;
    }
  });

  // Color scheme based on cognitive state
  const colors = useMemo(() => {
    const stateColors: Record<string, { primary: string; secondary: string; accent: string }> = {
      FLOW: { primary: "#00D4C8", secondary: "#8B5CF6", accent: "#10B981" },
      HESITATION: { primary: "#F59E0B", secondary: "#8B5CF6", accent: "#EAB308" },
      CONFUSION: { primary: "#EF4444", secondary: "#F97316", accent: "#DC2626" },
      CREATIVE: { primary: "#06B6D4", secondary: "#8B5CF6", accent: "#0EA5E9" },
      VERIFYING: { primary: "#3B82F6", secondary: "#8B5CF6", accent: "#2563EB" },
    };
    return stateColors[cognitiveState] || stateColors.FLOW;
  }, [cognitiveState]);

  return (
    <group ref={groupRef}>
      {/* Ambient light */}
      <ambientLight intensity={0.3} />
      <pointLight position={[5, 5, 5]} intensity={1} color={colors.primary} />
      <pointLight position={[-5, -5, -5]} intensity={0.5} color={colors.secondary} />

      {/* Neurons */}
      {neurons.map((neuron, i) => {
        const isActive =
          (activeComponent === "medulla" && neuron.region === "medulla") ||
          (activeComponent === "cortex" && neuron.region === "cortex") ||
          (activeComponent === "bridge" && neuron.region === "bridge");

        const color =
          neuron.region === "medulla"
            ? colors.primary
            : neuron.region === "cortex"
            ? colors.secondary
            : colors.accent;

        return (
          <Neuron
            key={i}
            position={neuron.pos}
            color={color}
            scale={0.8 + activity * 0.4}
            pulseSpeed={1 + entropy}
            isActive={isActive}
          />
        );
      })}

      {/* Synapses */}
      {synapses.map((synapse, i) => {
        const isActive =
          activeComponent === synapse.region ||
          (activeComponent === "bridge" && synapse.region !== "idle");

        return (
          <Synapse
            key={i}
            start={synapse.start}
            end={synapse.end}
            color={
              synapse.region === "medulla"
                ? colors.primary
                : synapse.region === "cortex"
                ? colors.secondary
                : colors.accent
            }
            activity={isActive ? activity : 0.1}
          />
        );
      })}

      {/* Central core glow */}
      <Float speed={2} rotationIntensity={0.2} floatIntensity={0.3}>
        <mesh position={[0, 0, 0]}>
          <sphereGeometry args={[0.3, 32, 32]} />
          <meshBasicMaterial color={colors.primary} transparent opacity={0.1 + activity * 0.2} />
        </mesh>
      </Float>
    </group>
  );
}

interface NeuralBrainProps {
  activeComponent?: "medulla" | "cortex" | "bridge" | "idle";
  cognitiveState?: string;
  entropy?: number;
  activity?: number;
  className?: string;
}

export function NeuralBrain({
  activeComponent = "medulla",
  cognitiveState = "FLOW",
  entropy = 1.0,
  activity = 0.5,
  className = "",
}: NeuralBrainProps) {
  const [webGLSupported, setWebGLSupported] = useState(true);
  const [hasError, setHasError] = useState(false);

  useEffect(() => {
    setWebGLSupported(isWebGLAvailable());
  }, []);

  // Show fallback if WebGL not supported or if there was a render error
  if (!webGLSupported || hasError) {
    return <WebGLFallback className={className} />;
  }

  return (
    <div className={`w-full h-full ${className}`}>
      <Canvas
        camera={{ position: [0, 0, 5], fov: 50 }}
        gl={{ antialias: true, alpha: true }}
        style={{ background: "transparent" }}
        onCreated={({ gl }) => {
          // Check for context loss
          gl.domElement.addEventListener("webglcontextlost", () => {
            setHasError(true);
          });
        }}
      >
        <NeuralNetwork
          activeComponent={activeComponent}
          cognitiveState={cognitiveState}
          entropy={entropy}
          activity={activity}
        />
        <OrbitControls
          enableZoom={false}
          enablePan={false}
          autoRotate
          autoRotateSpeed={0.5}
          minPolarAngle={Math.PI / 4}
          maxPolarAngle={Math.PI * 3 / 4}
        />
      </Canvas>
    </div>
  );
}
