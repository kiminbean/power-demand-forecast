/**
 * Alan Chat Screen - Page 2
 * Figma: iPhone 16 Pro - 12 (id: 2:219)
 * Design: Alan AI main screen with quick actions and chat input
 */

import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
  Dimensions,
  Animated,
} from 'react-native';
import { alanApi, AlanMessage, AlanResponse } from '../../services/alan/alanApi';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

// Design colors from Figma (Page 2: Alan Chat)
const colors = {
  primary: '#04265e',
  secondary: '#0048ff',
  background: '#d7dff0',      // Light lavender from Figma
  gradientStart: '#c5d0e8',
  gradientEnd: '#e0e6f2',
  white: '#ffffff',
  text: '#000000',
  textSecondary: '#666666',
  textMuted: '#999999',
  border: '#d0d5dd',
  inputBg: '#f5f7fa',
  buttonBlue: '#2563eb',
  robotFace: '#2d3a4f',       // Dark navy face
  robotEyes: '#00d4ff',       // Cyan eyes
};

// Quick action buttons from Figma design
const quickActions = [
  { id: 'register', label: '빠른 등록', icon: '' },
  { id: 'simulate', label: '수익 시뮬레이션', icon: '' },
  { id: 'trade', label: '스마트 거래', icon: '' },
  { id: 'plants', label: '내 발전소', icon: '' },
];

interface Props {
  onNavigate?: (screen: string, params?: any) => void;
}

// Alan Robot Avatar Component (matching Figma design - cat ears pointing up)
function AlanRobot() {
  const bounceAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    // Subtle floating animation
    const animation = Animated.loop(
      Animated.sequence([
        Animated.timing(bounceAnim, {
          toValue: -8,
          duration: 2000,
          useNativeDriver: Platform.OS !== 'web',
        }),
        Animated.timing(bounceAnim, {
          toValue: 0,
          duration: 2000,
          useNativeDriver: Platform.OS !== 'web',
        }),
      ])
    );
    animation.start();
    return () => animation.stop();
  }, []);

  return (
    <Animated.View style={[styles.alanRobot, { transform: [{ translateY: bounceAnim }] }]}>
      {/* Cat Ears (pointing up) */}
      <View style={styles.earsContainer}>
        <View style={styles.catEarLeft}>
          <View style={styles.catEarInnerLeft} />
        </View>
        <View style={styles.catEarRight}>
          <View style={styles.catEarInnerRight} />
        </View>
      </View>
      {/* Robot body */}
      <View style={styles.robotBody}>
        {/* Face area (dark) */}
        <View style={styles.robotFaceArea}>
          {/* Eyes */}
          <View style={styles.robotEyes}>
            <View style={styles.robotEye}>
              <View style={styles.robotPupil} />
            </View>
            <View style={styles.robotEye}>
              <View style={styles.robotPupil} />
            </View>
          </View>
          {/* Mouth */}
          <View style={styles.robotMouth}>
            <Text style={styles.robotMouthText}>‿</Text>
          </View>
        </View>
      </View>
      {/* Robot bottom (rounded) */}
      <View style={styles.robotBottom} />
    </Animated.View>
  );
}

// Quick Action Button Component
function QuickActionButton({
  label,
  icon,
  onPress,
}: {
  label: string;
  icon: string;
  onPress: () => void;
}) {
  return (
    <TouchableOpacity style={styles.quickActionBtn} onPress={onPress}>
      <Text style={styles.quickActionLabel}>{label}</Text>
    </TouchableOpacity>
  );
}

// Chat Message Component
function ChatMessage({ message }: { message: AlanMessage }) {
  const isUser = message.role === 'user';

  return (
    <View style={[styles.messageContainer, isUser ? styles.userMessage : styles.assistantMessage]}>
      <View style={[styles.messageBubble, isUser ? styles.userBubble : styles.assistantBubble]}>
        <Text style={[styles.messageText, isUser && styles.userMessageText]}>
          {message.content}
        </Text>
      </View>
    </View>
  );
}

export default function AlanChatScreen({ onNavigate }: Props) {
  const [messages, setMessages] = useState<AlanMessage[]>([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showChat, setShowChat] = useState(false);
  const scrollViewRef = useRef<ScrollView>(null);

  // Handle quick action press
  const handleQuickAction = (actionId: string) => {
    let prompt = '';
    switch (actionId) {
      case 'register':
        prompt = '발전소 빠른 등록 방법을 알려줘';
        break;
      case 'simulate':
        prompt = '수익 시뮬레이션을 해줘';
        break;
      case 'trade':
        prompt = '스마트 거래 현황을 보여줘';
        break;
      case 'plants':
        prompt = '내 발전소 목록을 보여줘';
        break;
    }
    if (prompt) {
      handleSendMessage(prompt);
    }
  };

  // Handle send message
  const handleSendMessage = async (text?: string) => {
    const messageText = text || inputText.trim();
    if (!messageText) return;

    setShowChat(true);
    setInputText('');

    // Add user message
    const userMessage: AlanMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: messageText,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);

    setIsLoading(true);

    try {
      // Call Alan API
      const response = await alanApi.sendMessage(messageText);

      // Add assistant message
      const assistantMessage: AlanMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.answer,
        timestamp: new Date(),
        action: response.action,
      };
      setMessages((prev) => [...prev, assistantMessage]);

      // Handle navigation action
      if (response.action?.type === 'navigate' && onNavigate) {
        setTimeout(() => {
          onNavigate(response.action!.target || '');
        }, 1500);
      }
    } catch (error) {
      console.error('[Alan] Error:', error);
      const errorMessage: AlanMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: '죄송합니다. 일시적인 오류가 발생했습니다. 다시 시도해 주세요.',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Auto scroll to bottom
  useEffect(() => {
    if (scrollViewRef.current && messages.length > 0) {
      setTimeout(() => {
        scrollViewRef.current?.scrollToEnd({ animated: true });
      }, 100);
    }
  }, [messages]);

  // Reset to main view
  const handleReset = () => {
    setShowChat(false);
    setMessages([]);
    alanApi.resetConversation();
  };

  // Render chat view
  if (showChat && messages.length > 0) {
    return (
      <KeyboardAvoidingView
        style={styles.container}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 90 : 0}
      >
        <View style={styles.chatHeader}>
          <TouchableOpacity onPress={handleReset} style={styles.backBtn}>
            <Text style={styles.backBtnText}>←</Text>
          </TouchableOpacity>
          <Text style={styles.chatHeaderTitle}>앨런 AI</Text>
          <View style={{ width: 40 }} />
        </View>

        <ScrollView
          ref={scrollViewRef}
          style={styles.chatMessages}
          contentContainerStyle={styles.chatMessagesContent}
          showsVerticalScrollIndicator={false}
        >
          {messages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))}
          {isLoading && (
            <View style={styles.loadingContainer}>
              <Text style={styles.loadingText}>앨런이 생각중...</Text>
            </View>
          )}
        </ScrollView>

        <View style={styles.chatInputContainer}>
          <View style={styles.chatInputWrapper}>
            <TextInput
              style={styles.chatInput}
              placeholder="메시지를 입력하세요..."
              placeholderTextColor={colors.textMuted}
              value={inputText}
              onChangeText={setInputText}
              onSubmitEditing={() => handleSendMessage()}
              returnKeyType="send"
            />
            <TouchableOpacity
              style={[styles.sendBtn, !inputText.trim() && styles.sendBtnDisabled]}
              onPress={() => handleSendMessage()}
              disabled={!inputText.trim()}
            >
              <Text style={styles.sendBtnText}>↑</Text>
            </TouchableOpacity>
          </View>
        </View>
      </KeyboardAvoidingView>
    );
  }

  // Render main Alan screen (Figma design)
  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      keyboardVerticalOffset={Platform.OS === 'ios' ? 90 : 0}
    >
      {/* Gradient Background */}
      <View style={styles.gradientBg}>
        {/* Alan Robot */}
        <View style={styles.robotContainer}>
          <AlanRobot />
        </View>

        {/* Quick Actions Grid */}
        <View style={styles.quickActionsContainer}>
          <View style={styles.quickActionsRow}>
            <QuickActionButton
              label="빠른 등록"
              icon=""
              onPress={() => handleQuickAction('register')}
            />
            <QuickActionButton
              label="수익 시뮬레이션"
              icon=""
              onPress={() => handleQuickAction('simulate')}
            />
          </View>
          <View style={styles.quickActionsRow}>
            <QuickActionButton
              label="스마트 거래"
              icon=""
              onPress={() => handleQuickAction('trade')}
            />
            <QuickActionButton
              label="내 발전소"
              icon=""
              onPress={() => handleQuickAction('plants')}
            />
          </View>
        </View>

        {/* Spacer */}
        <View style={styles.spacer} />

        {/* Chat Input Area */}
        <View style={styles.inputArea}>
          <View style={styles.inputWrapper}>
            <TextInput
              style={styles.input}
              placeholder="발전소 주변 실시간 기상 정보 알려줘"
              placeholderTextColor={colors.textMuted}
              value={inputText}
              onChangeText={setInputText}
              onSubmitEditing={() => handleSendMessage()}
              returnKeyType="send"
            />
            <TouchableOpacity
              style={styles.voiceBtn}
              onPress={() => handleSendMessage()}
            >
              <Text style={styles.voiceBtnIcon}>🎤</Text>
            </TouchableOpacity>
          </View>
        </View>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  gradientBg: {
    flex: 1,
    backgroundColor: colors.background,
  },

  // Robot Container
  robotContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingTop: 30,
    paddingBottom: 20,
  },
  alanRobot: {
    alignItems: 'center',
  },
  // Cat Ears (pointing up like in Figma)
  earsContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    width: 130,
    marginBottom: -15,
    zIndex: 1,
  },
  catEarLeft: {
    width: 0,
    height: 0,
    borderLeftWidth: 18,
    borderRightWidth: 18,
    borderBottomWidth: 35,
    borderLeftColor: 'transparent',
    borderRightColor: 'transparent',
    borderBottomColor: colors.white,
    marginRight: 40,
    transform: [{ rotate: '-15deg' }],
  },
  catEarInnerLeft: {
    position: 'absolute',
    top: 12,
    left: -8,
    width: 0,
    height: 0,
    borderLeftWidth: 8,
    borderRightWidth: 8,
    borderBottomWidth: 16,
    borderLeftColor: 'transparent',
    borderRightColor: 'transparent',
    borderBottomColor: '#f0f0f0',
  },
  catEarRight: {
    width: 0,
    height: 0,
    borderLeftWidth: 18,
    borderRightWidth: 18,
    borderBottomWidth: 35,
    borderLeftColor: 'transparent',
    borderRightColor: 'transparent',
    borderBottomColor: colors.white,
    marginLeft: 40,
    transform: [{ rotate: '15deg' }],
  },
  catEarInnerRight: {
    position: 'absolute',
    top: 12,
    left: -8,
    width: 0,
    height: 0,
    borderLeftWidth: 8,
    borderRightWidth: 8,
    borderBottomWidth: 16,
    borderLeftColor: 'transparent',
    borderRightColor: 'transparent',
    borderBottomColor: '#f0f0f0',
  },
  robotBody: {
    width: 130,
    height: 100,
    backgroundColor: colors.white,
    borderTopLeftRadius: 65,
    borderTopRightRadius: 65,
    borderBottomLeftRadius: 40,
    borderBottomRightRadius: 40,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 12,
    elevation: 5,
  },
  robotFaceArea: {
    width: 90,
    height: 60,
    backgroundColor: colors.robotFace,
    borderRadius: 30,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 5,
  },
  robotEyes: {
    flexDirection: 'row',
    gap: 16,
    marginTop: -2,
  },
  robotEye: {
    width: 20,
    height: 20,
    backgroundColor: colors.robotEyes,
    borderRadius: 10,
    alignItems: 'center',
    justifyContent: 'center',
  },
  robotPupil: {
    width: 6,
    height: 6,
    backgroundColor: '#ffffff',
    borderRadius: 3,
    marginTop: -3,
    marginLeft: 3,
  },
  robotMouth: {
    marginTop: 4,
  },
  robotMouthText: {
    fontSize: 14,
    color: colors.robotEyes,
    fontWeight: '400',
  },
  robotBottom: {
    width: 90,
    height: 35,
    backgroundColor: colors.white,
    borderBottomLeftRadius: 45,
    borderBottomRightRadius: 45,
    marginTop: -8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.08,
    shadowRadius: 8,
    elevation: 3,
  },

  // Quick Actions
  quickActionsContainer: {
    paddingHorizontal: 24,
    marginTop: 30,
  },
  quickActionsRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 12,
    marginBottom: 12,
  },
  quickActionBtn: {
    backgroundColor: colors.white,
    paddingVertical: 14,
    paddingHorizontal: 24,
    borderRadius: 24,
    borderWidth: 1,
    borderColor: colors.border,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  quickActionLabel: {
    fontSize: 15,
    fontWeight: '600',
    color: colors.text,
  },

  // Spacer
  spacer: {
    flex: 1,
  },

  // Input Area
  inputArea: {
    paddingHorizontal: 20,
    paddingBottom: Platform.OS === 'ios' ? 30 : 20,
  },
  inputWrapper: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.white,
    borderRadius: 28,
    paddingLeft: 20,
    paddingRight: 6,
    paddingVertical: 6,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 8,
    elevation: 3,
  },
  input: {
    flex: 1,
    fontSize: 15,
    color: colors.text,
    paddingVertical: 8,
  },
  voiceBtn: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: colors.buttonBlue,
    alignItems: 'center',
    justifyContent: 'center',
  },
  voiceBtnIcon: {
    fontSize: 20,
  },

  // Chat View Styles
  chatHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: colors.white,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  backBtn: {
    width: 40,
    height: 40,
    alignItems: 'center',
    justifyContent: 'center',
  },
  backBtnText: {
    fontSize: 24,
    color: colors.text,
  },
  chatHeaderTitle: {
    fontSize: 17,
    fontWeight: '600',
    color: colors.text,
  },
  chatMessages: {
    flex: 1,
    backgroundColor: colors.background,
  },
  chatMessagesContent: {
    padding: 16,
  },
  messageContainer: {
    marginBottom: 12,
  },
  userMessage: {
    alignItems: 'flex-end',
  },
  assistantMessage: {
    alignItems: 'flex-start',
  },
  messageBubble: {
    maxWidth: '80%',
    padding: 14,
    borderRadius: 18,
  },
  userBubble: {
    backgroundColor: colors.buttonBlue,
    borderBottomRightRadius: 4,
  },
  assistantBubble: {
    backgroundColor: colors.white,
    borderBottomLeftRadius: 4,
  },
  messageText: {
    fontSize: 15,
    color: colors.text,
    lineHeight: 22,
  },
  userMessageText: {
    color: colors.white,
  },
  loadingContainer: {
    alignItems: 'flex-start',
    marginTop: 8,
  },
  loadingText: {
    fontSize: 14,
    color: colors.textMuted,
    fontStyle: 'italic',
  },
  chatInputContainer: {
    padding: 12,
    backgroundColor: colors.white,
    borderTopWidth: 1,
    borderTopColor: colors.border,
  },
  chatInputWrapper: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.inputBg,
    borderRadius: 24,
    paddingLeft: 16,
    paddingRight: 4,
  },
  chatInput: {
    flex: 1,
    fontSize: 15,
    color: colors.text,
    paddingVertical: 12,
  },
  sendBtn: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: colors.buttonBlue,
    alignItems: 'center',
    justifyContent: 'center',
  },
  sendBtnDisabled: {
    backgroundColor: colors.textMuted,
  },
  sendBtnText: {
    fontSize: 18,
    color: colors.white,
    fontWeight: 'bold',
  },
});
