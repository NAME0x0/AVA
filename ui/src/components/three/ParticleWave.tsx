"use client";

import { useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";

interface ParticleFieldProps {
  count?: number;
  color?: string;
  activity?: number;
  entropy?: number;
}

function ParticleField({ count = 500, color = "#00D4C8", activity = 0.5, entropy = 1.0 }: ParticleFieldProps) {
  const pointsRef = useRef<THREE.Points>(null);

  // Generate particle positions
  const particles = useMemo(() => {
    const positions = new Float32Array(count * 3);
    const scales = new Float32Array(count);

    for (let i = 0; i < count; i++) {
      const i3 = i * 3;
      // Spread particles in a rectangular field
      positions[i3] = (Math.random() - 0.5) * 10;
      positions[i3 + 1] = (Math.random() - 0.5) * 4;
      positions[i3 + 2] = (Math.random() - 0.5) * 2;
      scales[i] = Math.random();
    }

    return { positions, scales };
  }, [count]);

  useFrame((state) => {
    if (pointsRef.current) {
      const positions = pointsRef.current.geometry.attributes.position.array as Float32Array;
      const time = state.clock.elapsedTime;

      for (let i = 0; i < count; i++) {
        const i3 = i * 3;
        const x = positions[i3];

        // Wave motion based on activity and entropy
        positions[i3 + 1] =
          Math.sin(x * 0.5 + time * (1 + activity)) * (0.3 + entropy * 0.3) +
          Math.sin(x * 0.3 + time * 0.5) * 0.2;

        // Slight z movement
        positions[i3 + 2] =
          Math.cos(x * 0.5 + time * 0.3) * 0.3 * activity;
      }

      pointsRef.current.geometry.attributes.position.needsUpdate = true;

      // Rotate slowly
      pointsRef.current.rotation.z = Math.sin(time * 0.1) * 0.05;
    }
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={count}
          array={particles.positions}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.03 + activity * 0.02}
        color={color}
        transparent
        opacity={0.6 + activity * 0.3}
        sizeAttenuation
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}

interface GlowingRingProps {
  radius: number;
  color: string;
  activity: number;
}

function GlowingRing({ radius, color, activity }: GlowingRingProps) {
  const ringRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (ringRef.current) {
      ringRef.current.rotation.z = state.clock.elapsedTime * 0.2;
      const scale = 1 + Math.sin(state.clock.elapsedTime * 2) * 0.05 * activity;
      ringRef.current.scale.setScalar(scale);
    }
  });

  return (
    <mesh ref={ringRef}>
      <ringGeometry args={[radius - 0.02, radius, 64]} />
      <meshBasicMaterial
        color={color}
        transparent
        opacity={0.3 + activity * 0.4}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

interface ParticleWaveProps {
  activity?: number;
  entropy?: number;
  cognitiveState?: string;
  className?: string;
}

export function ParticleWave({
  activity = 0.5,
  entropy = 1.0,
  cognitiveState = "FLOW",
  className = "",
}: ParticleWaveProps) {
  // Color based on cognitive state
  // Primary uses cyan (#00D4C8), secondary uses gold (#F5A623) for Cortex
  const colors = useMemo(() => {
    const stateColors: Record<string, { primary: string; secondary: string }> = {
      FLOW: { primary: "#00D4C8", secondary: "#F5A623" },
      HESITATION: { primary: "#F59E0B", secondary: "#EAB308" },
      CONFUSION: { primary: "#EF4444", secondary: "#F97316" },
      CREATIVE: { primary: "#06B6D4", secondary: "#F5A623" },
      VERIFYING: { primary: "#3B82F6", secondary: "#F5A623" },
    };
    return stateColors[cognitiveState] || stateColors.FLOW;
  }, [cognitiveState]);

  return (
    <div className={`w-full h-full ${className}`}>
      <Canvas
        camera={{ position: [0, 0, 5], fov: 60 }}
        gl={{ antialias: true, alpha: true }}
        style={{ background: "transparent" }}
      >
        <ambientLight intensity={0.2} />

        {/* Main particle field */}
        <ParticleField
          count={600}
          color={colors.primary}
          activity={activity}
          entropy={entropy}
        />

        {/* Secondary particle layer */}
        <ParticleField
          count={300}
          color={colors.secondary}
          activity={activity * 0.7}
          entropy={entropy * 0.8}
        />

        {/* Decorative rings */}
        <GlowingRing radius={2} color={colors.primary} activity={activity} />
        <GlowingRing radius={2.5} color={colors.secondary} activity={activity * 0.5} />
      </Canvas>
    </div>
  );
}
