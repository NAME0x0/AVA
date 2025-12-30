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
        // Primary palette - Theme-aware backgrounds via CSS variables
        neural: {
          void: "var(--neural-void)",
          surface: "var(--neural-surface)",
          elevated: "var(--neural-elevated)",
          hover: "var(--neural-hover)",
          border: "var(--neural-border)",
        },
        // Text hierarchy - Theme-aware
        text: {
          primary: "var(--text-primary)",
          secondary: "var(--text-secondary)",
          muted: "var(--text-muted)",
        },
        // Neural accent - Electric cyan (consistent across themes)
        accent: {
          primary: "#00D4C8",
          dim: "#00A89E",
          glow: "rgba(0, 212, 200, 0.3)",
          subtle: "rgba(0, 212, 200, 0.1)",
        },
        // Cognitive state colors
        state: {
          flow: "#00D4C8",
          hesitation: "#F5A623",
          confusion: "#EF4444",
          creative: "#8B5CF6",
          thinking: "#F5A623",
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
          surprise: "#F5A623",
          semantic: "#64FFAA",
        },
        // System status
        status: {
          success: "#10B981",
          warning: "#F5A623",
          error: "#EF4444",
          info: "#00D4C8",
        },
        // Cortex - Warm Gold (replacing purple)
        cortex: {
          active: "#F5A623",
          idle: "#C4841D",
          glow: "rgba(245, 166, 35, 0.3)",
        },
        // Medulla - Cool Teal (unified with accent)
        medulla: {
          active: "#00D4C8",
          processing: "#2DD4BF",
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
        "shake": "shake 0.5s ease-in-out",
        "golden-pulse": "golden-pulse 2s ease-in-out infinite",
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
        "shake": {
          "0%, 100%": { transform: "translateX(0)" },
          "10%, 30%, 50%, 70%, 90%": { transform: "translateX(-2px)" },
          "20%, 40%, 60%, 80%": { transform: "translateX(2px)" },
        },
        "golden-pulse": {
          "0%, 100%": { boxShadow: "0 0 5px rgba(245, 166, 35, 0.3)" },
          "50%": { boxShadow: "0 0 20px rgba(245, 166, 35, 0.6)" },
        },
      },
      boxShadow: {
        "neural": "0 0 30px rgba(0, 212, 200, 0.15)",
        "neural-strong": "0 0 50px rgba(0, 212, 200, 0.3)",
        "cortex": "0 0 20px rgba(245, 166, 35, 0.3)",
        "medulla": "0 0 20px rgba(0, 212, 200, 0.3)",
      },
      backdropBlur: {
        xs: "2px",
      },
      transitionProperty: {
        'theme': 'background-color, border-color, color, fill, stroke',
      },
    },
  },
  plugins: [],
};

export default config;
