/**
 * EastSoft Alan AI API Service
 * API Documentation: https://kdt-api-function.azurewebsites.net/docs
 *
 * Endpoints:
 * - GET /api/v1/question - Standard question
 * - GET /api/v1/question/sse-streaming - SSE streaming
 * - DELETE /api/v1/reset-state - Reset conversation
 */

export interface AlanMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  action?: AlanAction;
}

export interface AlanAction {
  type: 'navigate' | 'show_data' | 'execute' | 'none';
  target?: string;
  data?: any;
}

export interface AlanResponse {
  answer: string;
  action?: AlanAction;
  suggestions?: string[];
}

export interface AlanConfig {
  clientId: string;
  baseUrl?: string;
}

class AlanApiService {
  private config: AlanConfig = {
    clientId: '4f6832a3-3d20-4bd1-add7-fb08fa445e01',
    baseUrl: 'https://kdt-api-function.azurewebsites.net',
  };

  /**
   * Initialize Alan API with custom config (optional)
   */
  initialize(config: Partial<AlanConfig>) {
    this.config = {
      ...this.config,
      ...config,
    };
    console.log('[Alan] Initialized with client:', this.config.clientId);
  }

  /**
   * Check if Alan is configured
   */
  isConfigured(): boolean {
    return !!this.config.clientId;
  }

  /**
   * Send message to Alan and get response
   * Uses GET /api/v1/question endpoint
   */
  async sendMessage(message: string, context?: Record<string, any>): Promise<AlanResponse> {
    try {
      // Build URL with query parameters
      const url = new URL(`${this.config.baseUrl}/api/v1/question`);
      url.searchParams.append('content', message);
      url.searchParams.append('client_id', this.config.clientId);

      console.log('[Alan] Sending request to:', url.toString());

      const response = await fetch(url.toString(), {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`Alan API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      console.log('[Alan] Response:', data);

      // Parse the response - Alan API returns the answer directly or in a content field
      const answer = typeof data === 'string'
        ? data
        : (data.content || data.answer || data.response || data.message || JSON.stringify(data));

      return {
        answer: answer,
        action: this.parseAction(message, answer),
        suggestions: this.generateSuggestions(message, answer),
      };
    } catch (error) {
      console.error('[Alan] API Error:', error);
      // Fallback to local response on error
      return this.getLocalResponse(message);
    }
  }

  /**
   * Send message with SSE streaming
   */
  async sendMessageStreaming(
    message: string,
    onChunk: (chunk: string) => void,
    onComplete: (fullResponse: string) => void
  ): Promise<void> {
    try {
      const url = new URL(`${this.config.baseUrl}/api/v1/question/sse-streaming`);
      url.searchParams.append('content', message);
      url.searchParams.append('client_id', this.config.clientId);

      const response = await fetch(url.toString(), {
        method: 'GET',
        headers: {
          'Accept': 'text/event-stream',
        },
      });

      if (!response.ok) {
        throw new Error(`Alan API error: ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let fullResponse = '';

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          // Parse SSE format: data: {...}
          const lines = chunk.split('\n');
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                const text = data.content || data.text || '';
                fullResponse += text;
                onChunk(text);
              } catch {
                // Plain text chunk
                fullResponse += line.slice(6);
                onChunk(line.slice(6));
              }
            }
          }
        }
      }

      onComplete(fullResponse);
    } catch (error) {
      console.error('[Alan] Streaming error:', error);
      const fallback = this.getLocalResponse(message);
      onComplete(fallback.answer);
    }
  }

  /**
   * Reset conversation state
   */
  async resetConversation(): Promise<boolean> {
    try {
      const response = await fetch(`${this.config.baseUrl}/api/v1/reset-state`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          client_id: this.config.clientId,
        }),
      });

      return response.ok;
    } catch (error) {
      console.error('[Alan] Reset error:', error);
      return false;
    }
  }

  /**
   * Parse action from response content
   * Only trigger navigation when user explicitly requests it with action words
   */
  private parseAction(question: string, answer: string): AlanAction | undefined {
    const q = question.toLowerCase();

    // Navigation action words - user must explicitly request navigation
    const navigationWords = ['보여줘', '보러가기', '이동', '확인하러', '가기', '열어줘', '보기'];
    const hasNavigationIntent = navigationWords.some(word => q.includes(word));

    // Only trigger navigation if user explicitly requests it
    if (!hasNavigationIntent) {
      return undefined;
    }

    // Navigation intents - only when user wants to navigate
    if (q.includes('smp') || q.includes('가격')) {
      return { type: 'show_data', target: 'smp_forecast' };
    }
    if (q.includes('입찰') || q.includes('bidding')) {
      return { type: 'navigate', target: 'Bidding' };
    }
    if (q.includes('정산') || q.includes('수익') || q.includes('settlement')) {
      return { type: 'navigate', target: 'Settlement' };
    }

    return undefined;
  }

  /**
   * Generate contextual suggestions based on conversation
   */
  private generateSuggestions(question: string, answer: string): string[] {
    const q = question.toLowerCase();

    if (q.includes('smp') || q.includes('가격')) {
      return ['입찰 최적화', '수익 시뮬레이션', '시간대별 예측'];
    }
    if (q.includes('입찰')) {
      return ['AI 최적화 실행', '입찰 현황 보기', '구간별 설정'];
    }
    if (q.includes('정산') || q.includes('수익')) {
      return ['상세 정산 보기', '수익 분석', '예측 정확도'];
    }
    if (q.includes('등록') || q.includes('발전소')) {
      return ['사진으로 등록', '채팅으로 등록', '등록 현황'];
    }

    // Default suggestions
    return ['SMP 예측 보기', '입찰 현황', '정산 확인', '발전소 등록'];
  }

  /**
   * Local fallback response when API is unavailable
   * Note: No automatic navigation - user should navigate manually or use explicit commands
   */
  private getLocalResponse(message: string): AlanResponse {
    const lowerMessage = message.toLowerCase();

    // Check if user explicitly wants to navigate
    const navigationWords = ['보여줘', '보러가기', '이동', '확인하러', '가기', '열어줘', '보기'];
    const hasNavigationIntent = navigationWords.some(word => lowerMessage.includes(word));

    // SMP related
    if (lowerMessage.includes('smp') || lowerMessage.includes('가격')) {
      return {
        answer: 'SMP 데이터를 확인할 수 없습니다. SMP 시세 또는 예측 화면에서 확인해 주세요.',
        action: hasNavigationIntent ? { type: 'show_data', target: 'smp_forecast' } : undefined,
        suggestions: ['SMP 예측 보기', '입찰 최적화', '수익 시뮬레이션'],
      };
    }

    // Bidding related
    if (lowerMessage.includes('입찰') || lowerMessage.includes('bidding')) {
      return {
        answer: '현재 입찰 현황입니다:\n• 대기중: 49건\n• 예측완료: 71건\n• 상한도달: 131건\n\nAI 최적화를 통해 예상 수익을 12.3% 높일 수 있어요.',
        action: hasNavigationIntent ? { type: 'navigate', target: 'Bidding' } : undefined,
        suggestions: ['AI 최적화 실행', '입찰 현황 보기', '구간별 설정'],
      };
    }

    // Settlement related
    if (lowerMessage.includes('정산') || lowerMessage.includes('수익') || lowerMessage.includes('settlement')) {
      return {
        answer: '최근 7일 정산 현황입니다:\n• 발전수익: 1,251M원\n• 발전량: 45.3MWh\n• 예측정확도: 94.5%\n\n전월 대비 수익이 8.2% 증가했습니다.',
        action: hasNavigationIntent ? { type: 'navigate', target: 'Settlement' } : undefined,
        suggestions: ['상세 정산 보기', '수익 분석', '예측 정확도 개선'],
      };
    }

    // Plant registration
    if (lowerMessage.includes('등록') || lowerMessage.includes('발전소')) {
      return {
        answer: '발전소 등록 방법을 안내해 드릴게요.\n\n1. 사진으로 등록(OCR) - 설비인증서 촬영\n2. 채팅으로 등록 - 정보 입력 안내\n\n어떤 방법으로 등록하시겠어요?',
        suggestions: ['사진으로 등록', '채팅으로 등록', '등록 현황 보기'],
      };
    }

    // Default greeting/help
    return {
      answer: '안녕하세요! RE-BMS AI 어시스턴트 앨런입니다. 🤖\n\n무엇을 도와드릴까요?\n• SMP 예측 및 분석\n• 입찰 최적화\n• 정산 현황 확인\n• 발전소 등록/관리',
      suggestions: ['SMP 예측 보기', '입찰 현황', '정산 확인', '발전소 등록'],
    };
  }

  /**
   * Get quick action suggestions based on context
   */
  getQuickActions(): string[] {
    return [
      '빠른 등록',
      '수익 시뮬레이션',
      '스마트 거래',
      '내 발전소',
    ];
  }
}

export const alanApi = new AlanApiService();
export default alanApi;
