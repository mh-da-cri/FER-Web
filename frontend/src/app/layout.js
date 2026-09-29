import "./globals.css";

export const metadata = {
  title: "AI Camera — Phân Tích Biểu Cảm Khuôn Mặt",
  description:
    "Ứng dụng AI Camera phân tích biểu cảm khuôn mặt theo thời gian thực sử dụng MediaPipe và mô hình FER.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="vi">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta name="theme-color" content="#0a0e1a" />
      </head>
      <body>{children}</body>
    </html>
  );
}
