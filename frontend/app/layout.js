import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: '--font-sans' });
const mono = JetBrains_Mono({ subsets: ["latin"], variable: '--font-mono' });

export const metadata = {
  title: "DispatchCommand — RL Environment Monitor",
  description: "Real-time ambulance dispatch RL environment monitoring",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.variable} ${mono.variable} font-sans bg-[#04060f] text-slate-200 overflow-hidden`}>
        {children}
      </body>
    </html>
  );
}
