import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  description:
    'Early detection of Parkinson\u2019s disease from resting-state fMRI — OpenNeuro ds005892. Functional-connectivity deep learning, benchmarked honestly on 55 participants.',
  title: 'Parkinson\u2019s Early Detection — rs-fMRI | ds005892',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
