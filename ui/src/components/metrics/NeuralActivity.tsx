"use client";

import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface NeuralActivityProps {
  entropy: number;
  varentropy: number;
  className?: string;
}

export function NeuralActivity({ entropy, varentropy, className }: NeuralActivityProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [values, setValues] = useState<number[]>(Array(64).fill(0.5));
  const frameRef = useRef(0);

  useEffect(() => {
    // Update waveform values based on entropy/varentropy
    setValues((prev) => {
      const newValues = [...prev.slice(1)];
      const noise = Math.sin(frameRef.current * 0.1) * 0.2 + Math.cos(frameRef.current * 0.23) * 0.15;
      const value = Math.min(0.9, Math.max(0.1, (entropy / 5) * 0.6 + Math.sqrt(varentropy) * 0.3 + noise));
      newValues.push(value);
      frameRef.current++;
      return newValues;
    });
  }, [entropy, varentropy]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    // Safety check for browsers that may not support 2D context
    if (!ctx) {
      console.warn("NeuralActivity: Canvas 2D context not available");
      return;
    }

    const draw = () => {
      const { width, height } = canvas;
      ctx.clearRect(0, 0, width, height);

      // Background
      ctx.fillStyle = "rgba(14, 14, 20, 0.8)";
      ctx.roundRect(0, 0, width, height, 8);
      ctx.fill();

      // Draw waveform
      ctx.beginPath();
      ctx.strokeStyle = "#00D4C8";
      ctx.lineWidth = 2;

      values.forEach((v, i) => {
        const x = (i / values.length) * width;
        const y = height / 2 - (v - 0.5) * height * 0.8;
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();

      // Glow effect
      ctx.beginPath();
      ctx.strokeStyle = "rgba(0, 212, 200, 0.3)";
      ctx.lineWidth = 6;
      values.forEach((v, i) => {
        const x = (i / values.length) * width;
        const y = height / 2 - (v - 0.5) * height * 0.8;
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();

      // Center line
      ctx.beginPath();
      ctx.strokeStyle = "rgba(30, 30, 42, 0.8)";
      ctx.lineWidth = 1;
      ctx.moveTo(0, height / 2);
      ctx.lineTo(width, height / 2);
      ctx.stroke();
    };

    draw();
    const interval = setInterval(draw, 50);
    return () => clearInterval(interval);
  }, [values]);

  return (
    <canvas
      ref={canvasRef}
      width={268}
      height={60}
      className={cn("rounded-lg", className)}
    />
  );
}
