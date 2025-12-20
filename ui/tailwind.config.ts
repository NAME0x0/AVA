import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // Primary palette - Deep space blacks
        neural: {
          void: "#08080C",
          surface: "#0E0E14",
          elevated: "#16161E",
          hover: "#1E1E2A",
        },
        // Text hierarchy
        text: {
          primary: "#F0F0F5",
          secondary: "#A0A0AF",
          muted: "#64647B",
        },
        // Neural accent - Electric cyan/teal
        accent: {
          primary: "#00D4C8",
          dim: "#00968C",
          glow: "rgba(0, 212, 200, 0.3)",
        },
        // Cognitive state colors
        state: {
          flow: "#50C878",
          hesitation: "#FFC850",
          confusion: "#FF6464",
          creative: "#B464FF",
          thinking: "#6496FF",
        },
        // Sleep phases
        sleep: {
          drowsy: "#6464A0",
          light: "#50508C",
          deep: "#3C3C78",
          rem: "#7850B4",
        },
        // Memory/Learning
        memory: {
          episodic: "#64B4FF",
          surprise: "#FF9632",
          semantic: "#64FFAA",
        },
        // System status
        status: {
          success: "#50C878",
          warning: "#FFB432",
          error: "#FF5050",
        },
        // Cortex/Medulla specific
        cortex: {
          active: "#8B5CF6",
          idle: "#4C4C64",
        },
        medulla: {
          active: "#10B981",
          processing: "#34D399",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Menlo", "monospace"],
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "glow": "glow 2s ease-in-out infinite alternate",
        "wave": "wave 4s linear infinite",
        "neural-pulse": "neural-pulse 2s ease-in-out infinite",
        "fade-in": "fade-in 0.3s ease-out",
        "slide-up": "slide-up 0.4s ease-out",
        "slide-in-right": "slide-in-right 0.3s ease-out",
      },
      keyframes: {
        glow: {
          "0%": { boxShadow: "0 0 5px rgba(0, 212, 200, 0.3)" },
          "100%": { boxShadow: "0 0 20px rgba(0, 212, 200, 0.6)" },
        },
        wave: {
          "0%": { transform: "translateX(-100%)" },
          "100%": { transform: "translateX(100%)" },
        },
        "neural-pulse": {
          "0%, 100%": { opacity: "0.4" },
          "50%": { opacity: "1" },
        },
        "fade-in": {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        "slide-up": {
          "0%": { opacity: "0", transform: "translateY(10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "slide-in-right": {
          "0%": { opacity: "0", transform: "translateX(10px)" },
          "100%": { opacity: "1", transform: "translateX(0)" },
        },
      },
      boxShadow: {
        "neural": "0 0 30px rgba(0, 212, 200, 0.15)",
        "neural-strong": "0 0 50px rgba(0, 212, 200, 0.3)",
        "cortex": "0 0 20px rgba(139, 92, 246, 0.3)",
        "medulla": "0 0 20px rgba(16, 185, 129, 0.3)",
      },
      backdropBlur: {
        xs: "2px",
      },
    },
  },
  plugins: [],
};

export default config;
