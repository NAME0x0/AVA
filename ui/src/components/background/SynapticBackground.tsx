'use client';

import { useEffect, useRef, useCallback, memo } from 'react';
import { motion } from 'framer-motion';
import { useCoreStore } from '@/stores/core';

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  opacity: number;
  hue: number;
}

interface SynapticBackgroundProps {
  /** Enable particle system (can be disabled for performance) */
  particles?: boolean;
  /** Number of particles (default: 50) */
  particleCount?: number;
  /** Enable gradient mesh animation */
  gradientMesh?: boolean;
  /** Intensity of animations (0-1) */
  intensity?: number;
}

/**
 * SynapticBackground - Premium animated background with gradient mesh and particles
 *
 * Layers:
 * 1. Base: Deep void gradient
 * 2. Gradient Mesh: Slow-moving color blobs (CSS, GPU-accelerated)
 * 3. Particles: Floating constellation points (Canvas 2D)
 * 4. Noise overlay: Subtle texture (CSS)
 */
const SynapticBackground = memo(function SynapticBackground({
  particles = true,
  particleCount = 50,
  gradientMesh = true,
  intensity = 0.7,
}: SynapticBackgroundProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const particlesRef = useRef<Particle[]>([]);

  // Get cognitive state to adjust animation
  const cognitiveState = useCoreStore((state) => state.cognitiveState);
  const preferences = useCoreStore((state) => state.preferences);

  // Respect reduced motion preference
  const reduceMotion = preferences.theme?.reduceMotion || false;

  // Initialize particles
  const initParticles = useCallback((width: number, height: number) => {
    particlesRef.current = Array.from({ length: particleCount }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3,
      size: Math.random() * 2 + 1,
      opacity: Math.random() * 0.5 + 0.2,
      hue: Math.random() > 0.7 ? 45 : 175, // Gold or Cyan
    }));
  }, [particleCount]);

  // Animation loop for particles
  useEffect(() => {
    if (!particles || reduceMotion) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      initParticles(canvas.width, canvas.height);
    };
    resize();
    window.addEventListener('resize', resize);

    // Animation loop
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw connections between nearby particles
      ctx.strokeStyle = 'rgba(0, 212, 200, 0.05)';
      ctx.lineWidth = 0.5;

      for (let i = 0; i < particlesRef.current.length; i++) {
        for (let j = i + 1; j < particlesRef.current.length; j++) {
          const p1 = particlesRef.current[i];
          const p2 = particlesRef.current[j];
          const dx = p1.x - p2.x;
          const dy = p1.y - p2.y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < 150) {
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.globalAlpha = (1 - dist / 150) * 0.15 * intensity;
            ctx.stroke();
          }
        }
      }

      // Draw and update particles
      particlesRef.current.forEach((p) => {
        // Update position
        p.x += p.vx * intensity;
        p.y += p.vy * intensity;

        // Wrap around edges
        if (p.x < 0) p.x = canvas.width;
        if (p.x > canvas.width) p.x = 0;
        if (p.y < 0) p.y = canvas.height;
        if (p.y > canvas.height) p.y = 0;

        // Draw particle
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = p.hue === 45
          ? `hsla(${p.hue}, 100%, 65%, ${p.opacity * intensity})`
          : `hsla(${p.hue}, 100%, 50%, ${p.opacity * intensity})`;
        ctx.globalAlpha = 1;
        ctx.fill();

        // Add glow effect for larger particles
        if (p.size > 1.5) {
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.size * 3, 0, Math.PI * 2);
          const gradient = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.size * 3);
          gradient.addColorStop(0, p.hue === 45
            ? `hsla(${p.hue}, 100%, 65%, ${p.opacity * 0.3 * intensity})`
            : `hsla(${p.hue}, 100%, 50%, ${p.opacity * 0.3 * intensity})`);
          gradient.addColorStop(1, 'transparent');
          ctx.fillStyle = gradient;
          ctx.fill();
        }
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', resize);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [particles, reduceMotion, intensity, initParticles]);

  // Adjust animation based on cognitive state
  const isThinking = cognitiveState?.label === 'thinking' || cognitiveState?.label === 'hesitation';
  const meshSpeed = isThinking ? 15 : 25; // Faster when thinking

  return (
    <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none">
      {/* Layer 0: Deep void base */}
      <div className="absolute inset-0 bg-neural-void" />

      {/* Layer 1: Animated gradient mesh */}
      {gradientMesh && !reduceMotion && (
        <div className="absolute inset-0">
          {/* Primary cyan blob */}
          <motion.div
            className="absolute w-[600px] h-[600px] rounded-full opacity-30"
            style={{
              background: 'radial-gradient(circle, var(--accent-cyan) 0%, transparent 70%)',
              filter: 'blur(80px)',
            }}
            animate={{
              x: ['-10%', '30%', '10%', '-10%'],
              y: ['10%', '30%', '60%', '10%'],
            }}
            transition={{
              duration: meshSpeed,
              repeat: Infinity,
              ease: 'easeInOut',
            }}
          />

          {/* Secondary teal blob */}
          <motion.div
            className="absolute w-[500px] h-[500px] rounded-full opacity-20"
            style={{
              background: 'radial-gradient(circle, var(--accent-teal) 0%, transparent 70%)',
              filter: 'blur(100px)',
              right: '-5%',
            }}
            animate={{
              x: ['10%', '-20%', '5%', '10%'],
              y: ['60%', '20%', '40%', '60%'],
            }}
            transition={{
              duration: meshSpeed * 1.2,
              repeat: Infinity,
              ease: 'easeInOut',
            }}
          />

          {/* Warm gold accent blob */}
          <motion.div
            className="absolute w-[400px] h-[400px] rounded-full opacity-15"
            style={{
              background: 'radial-gradient(circle, var(--accent-gold) 0%, transparent 70%)',
              filter: 'blur(90px)',
            }}
            animate={{
              x: ['50%', '70%', '40%', '50%'],
              y: ['80%', '50%', '70%', '80%'],
            }}
            transition={{
              duration: meshSpeed * 1.5,
              repeat: Infinity,
              ease: 'easeInOut',
            }}
          />

          {/* Purple accent blob (subtle) */}
          <motion.div
            className="absolute w-[350px] h-[350px] rounded-full opacity-10"
            style={{
              background: 'radial-gradient(circle, var(--accent-purple) 0%, transparent 70%)',
              filter: 'blur(80px)',
              left: '60%',
              top: '10%',
            }}
            animate={{
              x: ['-10%', '10%', '-5%', '-10%'],
              y: ['0%', '20%', '10%', '0%'],
            }}
            transition={{
              duration: meshSpeed * 1.3,
              repeat: Infinity,
              ease: 'easeInOut',
            }}
          />
        </div>
      )}

      {/* Layer 2: Particle canvas */}
      {particles && !reduceMotion && (
        <canvas
          ref={canvasRef}
          className="absolute inset-0"
          style={{ opacity: intensity }}
        />
      )}

      {/* Layer 3: Subtle noise texture */}
      <div
        className="absolute inset-0 opacity-[0.015]"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E")`,
        }}
      />

      {/* Layer 4: Vignette overlay */}
      <div
        className="absolute inset-0"
        style={{
          background: 'radial-gradient(ellipse at center, transparent 0%, var(--neural-void) 100%)',
          opacity: 0.4,
        }}
      />

      {/* Thinking state pulse effect */}
      {isThinking && !reduceMotion && (
        <motion.div
          className="absolute inset-0"
          style={{
            background: 'radial-gradient(circle at center, var(--glow-gold) 0%, transparent 50%)',
          }}
          animate={{
            opacity: [0.1, 0.25, 0.1],
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
      )}
    </div>
  );
});

export default SynapticBackground;
