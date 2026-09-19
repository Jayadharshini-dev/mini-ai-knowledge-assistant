/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        control: {
          bg: "#f8fafc",
          panel: "#ffffff",
          border: "#e2e8f0",
          hover: "#f1f5f9",
          text: "#0f172a",
          muted: "#64748b",
          subtle: "#94a3b8",
          primary: "#2563eb",
          accent: "#059669",
          warning: "#d97706",
          danger: "#dc2626",
        },
      },
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'Courier New', 'monospace'],
      },
    },
  },
  plugins: [],
}
