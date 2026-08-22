/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'brand-dark': '#162530',
        'brand-muted': '#27404f',
        'brand-blue': '#02adca',
        'brand-teal': '#00cac3',
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'hero-glow': 'conic-gradient(from 180deg at 50% 50%, #02adca33 0deg, #00cac333 180deg, #02adca33 360deg)',
      }
    },
  },
  plugins: [],
}
