"use client";

import { useState, useRef, useEffect, useMemo } from "react";
import { sessionTimeline } from "@/utils/sessionTimeline";
import { EmotionChatEngine } from "@/utils/chatQueryEngine";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

const QUICK_PROMPTS = [
  "📊 Tóm tắt phiên làm việc",
  "😄 Vui vẻ chiếm bao nhiêu frame?",
  "⏱️ Giây thứ 10 cảm xúc gì?",
  "😴 Tôi có dấu hiệu mệt mỏi không?",
  "💡 Lời khuyên cải thiện tâm trạng",
];

export default function ChatPanel({ onJumpToTime }) {
  const [messages, setMessages] = useState([
    {
      id: "welcome",
      role: "assistant",
      content:
        "👋 Xin chào! Tôi là **AI Trợ Lý Phân Tích Cảm Xúc**.\n\n" +
        "Tôi có thể giúp bạn:\n" +
        "• ⏱️ **Tra cứu theo giây**: *'ở giây thứ 10 cảm xúc gì?'*\n" +
        "• 📊 **Thống kê số lượng**: *'bao nhiêu frame là vui?'*, *'tỷ lệ mệt mỏi?'*\n" +
        "• 🧠 **Phân tích chuyên sâu**: Đánh giá tâm trạng & đưa ra lời khuyên sức khỏe qua Gemini AI.\n\n" +
        "Hãy thử chọn gợi ý bên dưới hoặc nhập câu hỏi nhé!",
    },
  ]);

  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesContainerRef = useRef(null);
  const inputRef = useRef(null);

  // Khởi tạo Engine xử lý câu hỏi số liệu
  const chatEngine = useMemo(
    () => new EmotionChatEngine(sessionTimeline),
    []
  );

  // Cuộn NỘI BỘ trong khung tin nhắn thay vì cuộn toàn bộ màn hình
  useEffect(() => {
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTo({
        top: messagesContainerRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [messages, isLoading]);

  // ─── Gửi tin nhắn ───
  const handleSendMessage = async (customText = null) => {
    const text = (customText !== null ? customText : inputValue).trim();
    if (!text || isLoading) return;

    const userMsgId = Date.now().toString();
    const newMessages = [...messages, { id: userMsgId, role: "user", content: text }];
    setMessages(newMessages);
    setInputValue("");
    setIsLoading(true);

    try {
      // BƯỚC 1: Kiểm tra qua EmotionChatEngine (xử lý trực tiếp câu hỏi số liệu, 0ms latency)
      const directResult = chatEngine.processQuery(text);

      if (directResult && directResult.isDirectQuery) {
        setMessages((prev) => [
          ...prev,
          {
            id: (Date.now() + 1).toString(),
            role: "assistant",
            content: directResult.reply,
            source: "direct_engine",
            highlightTimestamp: directResult.highlightTimestamp,
          },
        ]);
        setIsLoading(false);
        return;
      }

      // BƯỚC 2: Nếu là câu hỏi phân tích tâm lý / tổng quan sâu -> Gửi backend Gemini API
      const stats = sessionTimeline.getStats();
      const timeline = sessionTimeline.getTimeline();
      const recentEvents = timeline.slice(-6).map((e) => ({
        timestamp: e.timestamp,
        emotion: e.dominantEmotion,
        reason: e.triggerReason,
      }));

      const payload = {
        message: text,
        timeline_summary: {
          ...stats,
          recent_events: recentEvents,
        },
        history: newMessages.slice(-6).map((m) => ({
          role: m.role,
          content: m.content,
        })),
      };

      const response = await fetch(`${API_BASE}/api/chat/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`Chat API error: ${response.status}`);
      }

      const data = await response.json();

      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: data.reply || "Không nhận được phản hồi từ AI.",
          source: data.source || "gemini",
        },
      ]);
    } catch (err) {
      console.error("[ChatPanel] Lỗi gửi chat:", err);
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content:
            "⚠️ Có lỗi khi kết nối tới máy chủ AI Chatbot. Vui lòng kiểm tra lại kết nối mạng hoặc thử lại sau.",
          source: "error",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // ─── Format nội dung tin nhắn và biến mốc thời gian [12.5s] thành nút bấm ───
  const renderMessageContent = (content) => {
    if (!content) return null;

    // Phân tách chuỗi theo mốc thời gian dạng [12.5s] hoặc [5s]
    const parts = content.split(/(\[\d+(?:\.\d+)?s\])/g);

    return parts.map((part, idx) => {
      const match = part.match(/\[(\d+(?:\.\d+)?)s\]/);
      if (match) {
        const seconds = parseFloat(match[1]);
        return (
          <button
            key={idx}
            className="chat-timestamp-badge"
            onClick={() => onJumpToTime?.(seconds)}
            title={`Nhấp để xem lại frame tại mốc ${seconds}s`}
          >
            ⏱️ {seconds}s
          </button>
        );
      }

      // Xử lý xuống dòng đơn giản
      return (
        <span key={idx}>
          {part.split("\n").map((line, lIdx, arr) => (
            <span key={lIdx}>
              {line}
              {lIdx < arr.length - 1 && <br />}
            </span>
          ))}
        </span>
      );
    });
  };

  return (
    <div className="glass-card chat-panel">
      {/* ─── Header Chatbot ─── */}
      <div className="chat-panel__header">
        <div className="chat-panel__title">
          <span className="chat-panel__avatar">🤖</span>
          <div>
            <div className="chat-panel__name">AI Trợ Lý Cảm Xúc</div>
            <div className="chat-panel__status">
              <span className="chat-status-dot" /> Sẵn sàng phân tích
            </div>
          </div>
        </div>

        <button
          className="btn-clear-chat"
          onClick={() =>
            setMessages([
              {
                id: "welcome-reset",
                role: "assistant",
                content: "Đã làm mới đoạn chat! Bạn có thể hỏi tôi bất kỳ điều gì về phiên làm việc nhé.",
              },
            ])
          }
          title="Xóa đoạn hội thoại"
        >
          🧹
        </button>
      </div>

      {/* ─── Khung hiển thị tin nhắn ─── */}
      <div className="chat-messages-container" ref={messagesContainerRef}>
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`chat-bubble-wrapper chat-bubble-wrapper--${msg.role}`}
          >
            {msg.role === "assistant" && (
              <div className="chat-avatar-mini">✨</div>
            )}
            <div className={`chat-bubble chat-bubble--${msg.role}`}>
              <div className="chat-bubble__text">
                {renderMessageContent(msg.content)}
              </div>
              {msg.source && msg.role === "assistant" && (
                <div className="chat-bubble__source">
                  {msg.source === "direct_engine"
                    ? "⚡ Truy vấn số liệu chính xác"
                    : msg.source === "gemini"
                    ? "🧠 Phân tích bởi Gemini LLM"
                    : "💡 Phân tích quy tắc"}
                </div>
              )}
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="chat-bubble-wrapper chat-bubble-wrapper--assistant">
            <div className="chat-avatar-mini">✨</div>
            <div className="chat-bubble chat-bubble--assistant chat-bubble--loading">
              <span className="chat-typing-dot" />
              <span className="chat-typing-dot" />
              <span className="chat-typing-dot" />
            </div>
          </div>
        )}
      </div>

      {/* ─── Quick Suggestion Chips ─── */}
      <div className="chat-quick-chips">
        {QUICK_PROMPTS.map((prompt, idx) => (
          <button
            key={idx}
            className="chip-btn"
            onClick={() => handleSendMessage(prompt)}
            disabled={isLoading}
          >
            {prompt}
          </button>
        ))}
      </div>

      {/* ─── Thanh nhập câu hỏi ─── */}
      <div className="chat-input-bar">
        <input
          ref={inputRef}
          type="text"
          className="chat-input"
          placeholder="Hỏi về giây thứ mấy, số lượng frame, tâm lý..."
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isLoading}
        />
        <button
          className="chat-send-btn"
          onClick={() => handleSendMessage()}
          disabled={!inputValue.trim() || isLoading}
          title="Gửi câu hỏi"
        >
          ➤
        </button>
      </div>
    </div>
  );
}
