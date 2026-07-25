import type { Metadata } from 'next';
import { Inter, JetBrains_Mono } from 'next/font/google';
import './globals.css';
import { Toaster } from 'react-hot-toast';

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-jetbrains-mono',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'HalluciSense — Confidence-Aware AI Assistant',
  description:
    'A confidence-aware hybrid framework for detecting and quantifying hallucinations in LLM responses. Know when to trust your AI.',
  keywords: ['hallucination detection', 'AI safety', 'LLM', 'ChatGPT', 'AI assistant'],
  openGraph: {
    title: 'HalluciSense',
    description: 'Know when to trust your AI.',
    type: 'website',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark" data-scroll-behavior="smooth">
      <body className={`${inter.variable} ${jetbrainsMono.variable} antialiased`}>
        <div className="bg-animated" aria-hidden="true" />
        {children}
        <Toaster
          position="bottom-right"
          toastOptions={{
            style: {
              background: '#0d1220',
              border: '1px solid rgba(255,255,255,0.08)',
              color: '#f1f5f9',
              borderRadius: '10px',
              fontSize: '14px',
            },
          }}
        />
      </body>
    </html>
  );
}
