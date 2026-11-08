import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Terrain AI | Image Classifier",
  description: "Classify landscape and scene images with a fine-tuned deep learning model.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
