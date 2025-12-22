"use client";

import { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";

interface StreamingTextProps {
  text: string;
  speed?: number; // ms per character
  onComplete?: () => void;
  className?: string;
  isStreaming?: boolean;
}

/**
 * StreamingText - Displays text with a typing animation effect
 *
 * Features:
 * - Configurable typing speed
 * - Natural variation in timing
 * - Cursor animation
 * - Immediate display when not streaming
 */
export function StreamingText({
  text,
  speed = 20,
  onComplete,
  className = "",
  isStreaming = true,
}: StreamingTextProps) {
  const [displayedText, setDisplayedText] = useState(isStreaming ? "" : text);
  const [isTyping, setIsTyping] = useState(isStreaming);
  const indexRef = useRef(0);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (!isStreaming) {
      setDisplayedText(text);
      setIsTyping(false);
      return;
    }

    indexRef.current = 0;
    setDisplayedText("");
    setIsTyping(true);

    const typeNextChar = () => {
      if (indexRef.current < text.length) {
        const char = text[indexRef.current];
        setDisplayedText(text.slice(0, indexRef.current + 1));
        indexRef.current++;

        // Add natural variation - longer pause after punctuation
        let delay = speed;
        if (['.', '!', '?'].includes(char)) {
          delay = speed * 4;
        } else if ([',', ';', ':'].includes(char)) {
          delay = speed * 2;
        } else {
          // Small random variation
          delay = speed + Math.random() * speed * 0.5;
        }

        timeoutRef.current = setTimeout(typeNextChar, delay);
      } else {
        setIsTyping(false);
        onComplete?.();
      }
    };

    timeoutRef.current = setTimeout(typeNextChar, speed);

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [text, speed, isStreaming, onComplete]);

  return (
    <span className={className}>
      {displayedText}
      {isTyping && (
        <motion.span
          className="inline-block w-0.5 h-4 ml-0.5 bg-primary"
          animate={{ opacity: [1, 0, 1] }}
          transition={{ duration: 0.8, repeat: Infinity }}
        />
      )}
    </span>
  );
}
