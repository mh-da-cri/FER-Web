/**
 * Quản lý dòng thời gian và thống kê cảm xúc theo phiên (Session Timeline Store).
 * Hỗ trợ lưu trữ chuỗi thời gian (time-series), tính toán phân bổ cảm xúc
 * và truy vấn mốc thời gian phục vụ cho UI thống kê và AI Chatbot.
 */

class SessionTimelineStore {
  constructor() {
    this.entries = [];
    this.elapsedBeforePause = 0;
    this.activeSince = null;
    this.subscribers = new Set();
    this.sessionActive = false;
  }

  /**
   * Bắt đầu một phiên ghi nhận mới hoặc tiếp tục phiên hiện tại.
   */
  start() {
    if (!this.sessionActive) {
      this.activeSince = Date.now();
      this.sessionActive = true;
    }
    this.notify();
  }

  /**
   * Tạm dừng ghi nhận.
   */
  pause() {
    if (this.sessionActive && this.activeSince !== null) {
      this.elapsedBeforePause += Date.now() - this.activeSince;
      this.activeSince = null;
    }
    this.sessionActive = false;
    this.notify();
  }

  /**
   * Đặt lại toàn bộ lịch sử phiên.
   */
  reset() {
    this.entries = [];
    this.elapsedBeforePause = 0;
    this.activeSince = this.sessionActive ? Date.now() : null;
    this.notify();
  }

  /**
   * Lấy số giây trôi qua từ khi bắt đầu phiên.
   * @returns {number} Số giây (làm tròn 1 chữ số thập phân)
   */
  getElapsedSeconds() {
    const activeDuration =
      this.elapsedBeforePause +
      (this.activeSince === null ? 0 : Date.now() - this.activeSince);
    return Math.round((activeDuration / 1000) * 10) / 10;
  }

  /**
   * Thêm một frame phân tích vào dòng thời gian.
   * @param {Object} payload Dữ liệu trả về từ API /api/analyze
   * @param {Object} metadata Thông tin bổ sung (lý do kích hoạt: 'emotion_changed', '10s_interval', 'manual')
   * @returns {Object} Entry vừa được thêm
   */
  addEntry(payload, metadata = {}) {
    if (!this.sessionActive) return null;

    const elapsed = this.getElapsedSeconds();
    const faces = payload?.faces || [];
    const mainFace = faces.find((f) => f.isFace);

    let dominantEmotion = null;
    let dominantConfidence = 0;
    const emotions = mainFace?.emotions || {};

    if (Object.keys(emotions).length > 0) {
      const sorted = Object.entries(emotions).sort((a, b) => b[1] - a[1]);
      dominantEmotion = sorted[0][0];
      dominantConfidence = sorted[0][1];
    }

    const entry = {
      frameIndex: this.entries.length + 1,
      timestamp: elapsed,
      wallClockTime: Date.now(),
      hasFace: !!mainFace,
      faceCount: faces.filter((f) => f.isFace).length,
      box: mainFace?.box || [],
      emotions: { ...emotions },
      fatigueScore: mainFace?.fatigue_score ?? 0,
      dominantEmotion,
      dominantConfidence,
      triggerReason: metadata.reason || "interval",
    };

    this.entries.push(entry);
    this.notify(entry);
    return entry;
  }

  /**
   * Lấy danh sách toàn bộ dòng thời gian.
   * @returns {Array} Mảng các entry
   */
  getTimeline() {
    return [...this.entries];
  }

  /**
   * Lấy N frame gần nhất (dùng để vẽ biểu đồ mini hoặc trượt cửa sổ).
   * @param {number} n Số lượng frame gần nhất
   */
  getRecentEntries(n = 30) {
    return this.entries.slice(-n);
  }

  /**
   * Tính toán thống kê toàn diện của toàn phiên.
   * @returns {Object} Bản tóm tắt thống kê
   */
  getStats() {
    const totalFrames = this.entries.length;
    const faceFrames = this.entries.filter((e) => e.hasFace).length;
    const duration = this.getElapsedSeconds();

    const emotionCounts = {};
    let validEmotionFrames = 0;

    for (const entry of this.entries) {
      if (entry.dominantEmotion) {
        emotionCounts[entry.dominantEmotion] = (emotionCounts[entry.dominantEmotion] || 0) + 1;
        validEmotionFrames++;
      }
    }

    const emotionPercentages = {};
    for (const [emotion, count] of Object.entries(emotionCounts)) {
      emotionPercentages[emotion] =
        validEmotionFrames > 0
          ? Math.round((count / validEmotionFrames) * 1000) / 10
          : 0;
    }

    const fatigueFrames = this.entries.filter((e) => e.hasFace && Number.isFinite(e.fatigueScore));
    const fatiguePercentage = fatigueFrames.length > 0
      ? (fatigueFrames.reduce((sum, e) => sum + e.fatigueScore, 0) / fatigueFrames.length) * 100
      : 0;

    let dominantOverall = null;
    let highestCount = 0;
    for (const [emotion, count] of Object.entries(emotionCounts)) {
      if (count > highestCount) {
        highestCount = count;
        dominantOverall = emotion;
      }
    }

    return {
      totalFrames,
      faceFrames,
      duration,
      emotionCounts,
      emotionPercentages,
      fatiguePercentage: Math.round(fatiguePercentage * 10) / 10,
      dominantOverall: dominantOverall
        ? {
            name: dominantOverall,
            count: highestCount,
            percentage: emotionPercentages[dominantOverall] || 0,
          }
        : null,
    };
  }

  /**
   * Tìm entry gần với giây thứ targetSecond nhất.
   * Rất hữu ích cho câu hỏi: "Ở giây thứ mấy là cảm xúc gì?"
   * @param {number} targetSecond Giây cần tìm
   * @param {number} maxDelta Độ lệch tối đa cho phép (mặc định 1.5 giây)
   */
  getEntryAtSecond(targetSecond, maxDelta = 1.5) {
    if (this.entries.length === 0) return null;

    let closest = null;
    let minDiff = Infinity;

    for (const entry of this.entries) {
      const diff = Math.abs(entry.timestamp - targetSecond);
      if (diff < minDiff) {
        minDiff = diff;
        closest = entry;
      }
    }

    if (minDiff <= maxDelta) {
      return { entry: closest, delta: minDiff };
    }
    return null;
  }

  /**
   * Lọc các entry trong khoảng thời gian [startSec, endSec].
   * @param {number} startSec Giây bắt đầu
   * @param {number} endSec Giây kết thúc
   */
  getEntriesInRange(startSec, endSec) {
    return this.entries.filter(
      (e) => e.timestamp >= startSec && e.timestamp <= endSec
    );
  }

  /**
   * Đăng ký callback khi có entry mới hoặc khi reset.
   * @param {Function} listener
   * @returns {Function} Hàm hủy đăng ký
   */
  subscribe(listener) {
    this.subscribers.add(listener);
    return () => {
      this.subscribers.delete(listener);
    };
  }

  notify(newEntry = null) {
    const stats = this.getStats();
    for (const listener of this.subscribers) {
      try {
        listener({
          timeline: this.entries,
          stats,
          newEntry,
          duration: this.getElapsedSeconds(),
        });
      } catch (err) {
        console.error("[SessionTimelineStore] Subscriber error:", err);
      }
    }
  }
}

// Singleton instance dùng chung cho toàn bộ ứng dụng
export const sessionTimeline = new SessionTimelineStore();

export function formatSessionDuration(totalSeconds) {
  const seconds = Math.max(0, Math.floor(totalSeconds));
  if (seconds < 60) return `${seconds}s`;
  return `${Math.floor(seconds / 60)} phút ${seconds % 60} giây`;
}
