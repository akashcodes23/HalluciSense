import type { Metadata } from "next";
import { Inter, Space_Grotesk, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { Providers } from "./providers";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-space-grotesk",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "HalluciSense — Detect Hallucinations with Scientific Confidence",
  description:
    "Confidence-aware AI verification powered by a three-pillar hallucination detection framework: Evidence Support, Model Uncertainty, and Generation Consistency.",
  keywords: [
    "hallucination detection",
    "AI safety",
    "LLM verification",
    "factual accuracy",
    "AI confidence",
  ],
  openGraph: {
    title: "HalluciSense",
    description:
      "Detect hallucinations with scientific confidence. Three-pillar AI verification framework.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${inter.variable} ${spaceGrotesk.variable} ${jetbrainsMono.variable} antialiased`}
      >
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
