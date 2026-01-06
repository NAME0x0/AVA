import type { Metadata, Viewport } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { ThemeProvider } from "@/providers/ThemeProvider";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: "AVA - Neural Second Brain",
  description: "Adaptive Virtual Agent - Biomimetic Neural Architecture with Cortex-Medulla dual-brain system",
  keywords: ["AI", "assistant", "neural", "cortex", "medulla", "AVA", "second brain", "cognitive", "reasoning"],
  authors: [{ name: "Muhammad Afsah Mumtaz" }],
  icons: {
    icon: [
      { url: "/favicon.svg", type: "image/svg+xml" },
      { url: "/favicon-32x32.png", sizes: "32x32", type: "image/png" },
      { url: "/favicon-16x16.png", sizes: "16x16", type: "image/png" },
    ],
    apple: [
      { url: "/apple-touch-icon.png", sizes: "180x180", type: "image/png" },
    ],
  },
  manifest: "/site.webmanifest",
  openGraph: {
    title: "AVA - Neural Second Brain",
    description: "Adaptive Virtual Agent - Biomimetic Neural Architecture",
    images: [{ url: "/og-image.png", width: 1200, height: 630, alt: "AVA Neural Interface" }],
    type: "website",
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#FAFBFC" },
    { media: "(prefers-color-scheme: dark)", color: "#08080C" },
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning className="h-full">
      <body
        className={`${inter.variable} ${jetbrainsMono.variable} font-sans h-full bg-neural-void overflow-hidden`}
        style={{ backgroundColor: '#08080C' }} // Ensure solid background even before CSS loads
      >
        <ThemeProvider defaultMode="dark">
          <ErrorBoundary>{children}</ErrorBoundary>
        </ThemeProvider>
      </body>
    </html>
  );
}
