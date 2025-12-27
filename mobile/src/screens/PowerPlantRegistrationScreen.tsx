/**
 * Power Plant Registration Screen
 * eXeco Mobile v6.2.0
 * Register small-scale solar/wind/ESS plants
 */

import React, { useState, useMemo } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
  Platform,
  Alert,
  ActivityIndicator,
  Modal,
} from 'react-native';
import { WebView } from 'react-native-webview';
import {
  PowerPlant,
  PlantType,
  ContractType,
  RoofDirection,
  PLANT_TYPE_LABELS,
  CONTRACT_TYPE_LABELS,
  ROOF_DIRECTION_LABELS,
} from '../types/powerPlant';
import {
  calculateEfficiency,
  estimateDailyGeneration,
  estimateMonthlyGeneration,
  estimateRevenue,
  getEfficiencyStatus,
  formatRevenue,
  WeatherCondition,
} from '../utils/powerPlantUtils';
import apiService from '../services/api';

// Colors matching app theme
const colors = {
  primary: '#04265e',
  secondary: '#0048ff',
  background: '#f8fafc',
  card: '#ffffff',
  text: '#1e293b',
  textMuted: '#64748b',
  border: '#e2e8f0',
  green: '#22c55e',
  orange: '#f97316',
  red: '#ef4444',
  blue: '#3b82f6',
};

interface PowerPlantRegistrationScreenProps {
  onClose: () => void;
  onSave: (plant: PowerPlant) => void;
  editPlant?: PowerPlant; // For editing existing plant
  currentSmpPrice?: number; // Current SMP for revenue estimation
  currentWeather?: WeatherCondition;
}

export default function PowerPlantRegistrationScreen({
  onClose,
  onSave,
  editPlant,
  currentSmpPrice = 80, // Default SMP
  currentWeather = 'clear',
}: PowerPlantRegistrationScreenProps) {
  // Form state
  const [plantName, setPlantName] = useState(editPlant?.name || '');
  const [plantType, setPlantType] = useState<PlantType>(editPlant?.type || 'solar');
  const [capacity, setCapacity] = useState(editPlant?.capacity?.toString() || '3');
  const [installYear, setInstallYear] = useState(
    editPlant ? new Date(editPlant.installDate).getFullYear() : new Date().getFullYear()
  );
  const [installMonth, setInstallMonth] = useState(
    editPlant ? new Date(editPlant.installDate).getMonth() + 1 : 1
  );
  const [contractType, setContractType] = useState<ContractType>(
    editPlant?.contractType || 'ppa'
  );
  const [roofDirection, setRoofDirection] = useState<RoofDirection>(
    editPlant?.roofDirection || 'south'
  );
  const [address, setAddress] = useState(editPlant?.location?.address || '');

  // UI state
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [showAddressModal, setShowAddressModal] = useState(false);

  // Daum Postcode API HTML - iOS 호환 버전
  const daumPostcodeHtml = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>주소 검색</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    html, body { width: 100%; height: 100%; overflow: hidden; background: #fff; }
    #wrap { width: 100%; height: 100%; }
  </style>
</head>
<body>
  <div id="wrap"></div>
  <script src="https://t1.daumcdn.net/mapjsapi/bundle/postcode/prod/postcode.v2.js"></script>
  <script>
    function sendToApp(address, zonecode) {
      try {
        var msg = JSON.stringify({ type: 'ADDRESS', address: address, zonecode: zonecode });
        if (window.ReactNativeWebView && window.ReactNativeWebView.postMessage) {
          window.ReactNativeWebView.postMessage(msg);
        } else if (window.webkit && window.webkit.messageHandlers && window.webkit.messageHandlers.ReactNativeWebView) {
          window.webkit.messageHandlers.ReactNativeWebView.postMessage(msg);
        }
      } catch(e) {
        console.error('sendToApp error:', e);
      }
    }

    new daum.Postcode({
      oncomplete: function(data) {
        var fullAddr = data.roadAddress || data.jibunAddress || data.address;
        if (data.buildingName && data.buildingName !== '') {
          fullAddr += ' (' + data.buildingName + ')';
        }
        sendToApp(fullAddr, data.zonecode);
      },
      width: '100%',
      height: '100%'
    }).embed(document.getElementById('wrap'));
  </script>
</body>
</html>
  `;

  // Handle address selection from WebView
  const handleAddressMessage = (event: any) => {
    try {
      const rawData = event?.nativeEvent?.data;
      if (!rawData || typeof rawData !== 'string') return;

      const data = JSON.parse(rawData);
      console.log('Address message received:', data);

      if (data.type === 'ADDRESS' && data.address) {
        setAddress(data.address);
        setTimeout(() => setShowAddressModal(false), 100);
      }
    } catch (e) {
      // 다른 WebView 내부 메시지 무시
      console.log('Non-address message:', e);
    }
  };

  // Calculate estimates based on form values
  const estimates = useMemo(() => {
    const capacityNum = parseFloat(capacity) || 0;
    const installDate = `${installYear}-${String(installMonth).padStart(2, '0')}-01`;
    const efficiency = calculateEfficiency(installDate);
    const dailyKwh = estimateDailyGeneration(capacityNum, efficiency, currentWeather, roofDirection);
    const monthlyKwh = estimateMonthlyGeneration(capacityNum, efficiency, roofDirection);
    const monthlyRevenue = estimateRevenue(dailyKwh, currentSmpPrice, contractType);
    const efficiencyStatus = getEfficiencyStatus(efficiency);

    return {
      efficiency,
      efficiencyStatus,
      dailyKwh,
      monthlyKwh,
      monthlyRevenue,
    };
  }, [capacity, installYear, installMonth, roofDirection, currentWeather, currentSmpPrice, contractType]);

  // Validation
  const validate = (): boolean => {
    const newErrors: Record<string, string> = {};

    if (!plantName.trim()) {
      newErrors.name = '발전소 이름을 입력하세요';
    }

    const capacityNum = parseFloat(capacity);
    if (isNaN(capacityNum) || capacityNum <= 0) {
      newErrors.capacity = '유효한 용량을 입력하세요';
    }

    if (installYear < 2000 || installYear > new Date().getFullYear()) {
      newErrors.installYear = '유효한 설치 연도를 입력하세요';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  // Submit handler
  const handleSubmit = async () => {
    if (!validate()) return;

    setIsSubmitting(true);

    try {
      const installDate = `${installYear}-${String(installMonth).padStart(2, '0')}-01`;
      const now = new Date().toISOString();

      const plantData: PowerPlant = {
        id: editPlant?.id || `plant_${Date.now()}`,
        name: plantName.trim(),
        type: plantType,
        capacity: parseFloat(capacity),
        installDate,
        contractType,
        location: {
          address: address.trim(),
        },
        roofDirection,
        createdAt: editPlant?.createdAt || now,
        updatedAt: now,
      };

      // Call API to save
      try {
        if (editPlant) {
          await apiService.updatePowerPlant(plantData.id, plantData);
        } else {
          await apiService.createPowerPlant(plantData);
        }
      } catch (apiError) {
        console.log('API save failed, using local storage:', apiError);
        // Fallback: save locally (handled by parent)
      }

      onSave(plantData);

      const message = editPlant ? '발전소가 수정되었습니다.' : '발전소가 등록되었습니다.';
      if (Platform.OS === 'web') {
        window.alert(message);
      } else {
        Alert.alert('완료', message);
      }
    } catch (error) {
      const errorMsg = '저장 중 오류가 발생했습니다.';
      if (Platform.OS === 'web') {
        window.alert(errorMsg);
      } else {
        Alert.alert('오류', errorMsg);
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  // Generate year options
  const yearOptions = Array.from(
    { length: 30 },
    (_, i) => new Date().getFullYear() - i
  );

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={onClose} style={styles.backButton}>
          <Text style={styles.backButtonText}>{'<'} 취소</Text>
        </TouchableOpacity>
        <Text style={styles.headerTitle}>
          {editPlant ? '발전소 수정' : '내 발전소 등록'}
        </Text>
        <View style={styles.headerRight} />
      </View>

      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        {/* Plant Name */}
        <View style={styles.formGroup}>
          <Text style={styles.label}>발전소 이름 *</Text>
          <TextInput
            style={[styles.input, errors.name ? styles.inputError : undefined]}
            value={plantName}
            onChangeText={setPlantName}
            placeholder="예: 우리집 태양광 1호"
            placeholderTextColor={colors.textMuted}
          />
          {errors.name && <Text style={styles.errorText}>{errors.name}</Text>}
        </View>

        {/* Plant Type */}
        <View style={styles.formGroup}>
          <Text style={styles.label}>설비 유형</Text>
          <View style={styles.segmentedControl}>
            {(Object.keys(PLANT_TYPE_LABELS) as PlantType[]).map((type) => (
              <TouchableOpacity
                key={type}
                style={[
                  styles.segmentButton,
                  plantType === type && styles.segmentButtonActive,
                ]}
                onPress={() => setPlantType(type)}
              >
                <Text
                  style={[
                    styles.segmentButtonText,
                    plantType === type && styles.segmentButtonTextActive,
                  ]}
                >
                  {PLANT_TYPE_LABELS[type].icon} {PLANT_TYPE_LABELS[type].label}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        {/* Capacity */}
        <View style={styles.formGroup}>
          <Text style={styles.label}>설비 용량 *</Text>
          <View style={styles.inputRow}>
            <TextInput
              style={[styles.input, styles.inputWithUnit, errors.capacity ? styles.inputError : undefined]}
              value={capacity}
              onChangeText={setCapacity}
              keyboardType="numeric"
              placeholder="3"
              placeholderTextColor={colors.textMuted}
            />
            <Text style={styles.unitText}>kW</Text>
          </View>
          {errors.capacity && <Text style={styles.errorText}>{errors.capacity}</Text>}
          <Text style={styles.helpText}>
            가정용: 3~10kW | 상업용: 10~100kW | 대규모: 100kW 이상
          </Text>
        </View>

        {/* Install Date */}
        <View style={styles.formGroup}>
          <Text style={styles.label}>설치일</Text>
          <View style={styles.dateRow}>
            <View style={styles.dateInput}>
              <Text style={styles.dateLabel}>년</Text>
              <View style={styles.pickerContainer}>
                <TouchableOpacity
                  style={styles.pickerButton}
                  onPress={() => setInstallYear(Math.max(2000, installYear - 1))}
                >
                  <Text style={styles.pickerButtonText}>-</Text>
                </TouchableOpacity>
                <Text style={styles.pickerValue}>{installYear}</Text>
                <TouchableOpacity
                  style={styles.pickerButton}
                  onPress={() => setInstallYear(Math.min(new Date().getFullYear(), installYear + 1))}
                >
                  <Text style={styles.pickerButtonText}>+</Text>
                </TouchableOpacity>
              </View>
            </View>
            <View style={styles.dateInput}>
              <Text style={styles.dateLabel}>월</Text>
              <View style={styles.pickerContainer}>
                <TouchableOpacity
                  style={styles.pickerButton}
                  onPress={() => setInstallMonth(Math.max(1, installMonth - 1))}
                >
                  <Text style={styles.pickerButtonText}>-</Text>
                </TouchableOpacity>
                <Text style={styles.pickerValue}>{installMonth}</Text>
                <TouchableOpacity
                  style={styles.pickerButton}
                  onPress={() => setInstallMonth(Math.min(12, installMonth + 1))}
                >
                  <Text style={styles.pickerButtonText}>+</Text>
                </TouchableOpacity>
              </View>
            </View>
          </View>
          {errors.installYear && <Text style={styles.errorText}>{errors.installYear}</Text>}
        </View>

        {/* Efficiency Display */}
        <View style={styles.efficiencyCard}>
          <Text style={styles.efficiencyLabel}>현재 효율</Text>
          <View style={styles.efficiencyRow}>
            <Text style={[styles.efficiencyValue, { color: estimates.efficiencyStatus.color }]}>
              {estimates.efficiencyStatus.text}
            </Text>
            <Text style={styles.efficiencyYear}>
              ({estimates.efficiencyStatus.yearText})
            </Text>
          </View>
          <Text style={styles.efficiencyHelp}>
            설치 후 1년차 3%, 이후 매년 0.6%씩 효율 감소
          </Text>
        </View>

        {/* Contract Type */}
        <View style={styles.formGroup}>
          <Text style={styles.label}>계약 유형</Text>
          <View style={styles.radioGroup}>
            {(Object.keys(CONTRACT_TYPE_LABELS) as ContractType[]).map((type) => (
              <TouchableOpacity
                key={type}
                style={styles.radioItem}
                onPress={() => setContractType(type)}
              >
                <View style={[styles.radio, contractType === type && styles.radioActive]}>
                  {contractType === type && <View style={styles.radioDot} />}
                </View>
                <View style={styles.radioContent}>
                  <Text style={styles.radioLabel}>
                    {CONTRACT_TYPE_LABELS[type].label}
                  </Text>
                  <Text style={styles.radioDesc}>
                    {CONTRACT_TYPE_LABELS[type].description}
                  </Text>
                </View>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        {/* Roof Direction (only for solar) */}
        {plantType === 'solar' && (
          <View style={styles.formGroup}>
            <Text style={styles.label}>지붕 방향 (선택)</Text>
            <View style={styles.directionRow}>
              {(Object.keys(ROOF_DIRECTION_LABELS) as RoofDirection[]).map((dir) => (
                <TouchableOpacity
                  key={dir}
                  style={[
                    styles.directionButton,
                    roofDirection === dir && styles.directionButtonActive,
                  ]}
                  onPress={() => setRoofDirection(dir)}
                >
                  <Text
                    style={[
                      styles.directionButtonText,
                      roofDirection === dir && styles.directionButtonTextActive,
                    ]}
                  >
                    {ROOF_DIRECTION_LABELS[dir]}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>
        )}

        {/* Address */}
        <View style={styles.formGroup}>
          <Text style={styles.label}>주소 (선택)</Text>
          <View style={styles.addressRow}>
            <TextInput
              style={[styles.input, styles.addressInput]}
              value={address}
              onChangeText={setAddress}
              placeholder="주소 검색을 눌러주세요"
              placeholderTextColor={colors.textMuted}
              editable={false}
            />
            <TouchableOpacity
              style={styles.addressSearchButton}
              onPress={() => setShowAddressModal(true)}
            >
              <Text style={styles.addressSearchButtonText}>🔍 검색</Text>
            </TouchableOpacity>
          </View>
          {address ? (
            <TouchableOpacity onPress={() => setAddress('')}>
              <Text style={styles.clearAddressText}>✕ 주소 삭제</Text>
            </TouchableOpacity>
          ) : null}
        </View>

        {/* Estimates Summary */}
        <View style={styles.estimatesCard}>
          <Text style={styles.estimatesTitle}>예상 발전량 및 수익</Text>
          <View style={styles.estimatesGrid}>
            <View style={styles.estimateItem}>
              <Text style={styles.estimateLabel}>일일 발전량</Text>
              <Text style={styles.estimateValue}>{estimates.dailyKwh} kWh</Text>
            </View>
            <View style={styles.estimateItem}>
              <Text style={styles.estimateLabel}>월간 발전량</Text>
              <Text style={styles.estimateValue}>{estimates.monthlyKwh} kWh</Text>
            </View>
            <View style={styles.estimateItem}>
              <Text style={styles.estimateLabel}>예상 월간 수익</Text>
              <Text style={[styles.estimateValue, { color: colors.green }]}>
                {formatRevenue(estimates.monthlyRevenue)}
              </Text>
            </View>
            <View style={styles.estimateItem}>
              <Text style={styles.estimateLabel}>기준 SMP</Text>
              <Text style={styles.estimateValue}>{currentSmpPrice}원/kWh</Text>
            </View>
          </View>
          <Text style={styles.estimatesHelp}>
            * 실제 발전량은 날씨, 계절에 따라 변동될 수 있습니다
          </Text>
        </View>

        {/* Submit Button */}
        <View style={styles.buttonGroup}>
          <TouchableOpacity
            style={styles.cancelButton}
            onPress={onClose}
            disabled={isSubmitting}
          >
            <Text style={styles.cancelButtonText}>취소</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.submitButton, isSubmitting && styles.submitButtonDisabled]}
            onPress={handleSubmit}
            disabled={isSubmitting}
          >
            {isSubmitting ? (
              <ActivityIndicator color="#ffffff" size="small" />
            ) : (
              <Text style={styles.submitButtonText}>
                {editPlant ? '수정' : '등록'}
              </Text>
            )}
          </TouchableOpacity>
        </View>

        {/* Bottom padding */}
        <View style={{ height: 40 }} />
      </ScrollView>

      {/* Address Search Modal */}
      <Modal
        visible={showAddressModal}
        animationType="slide"
        onRequestClose={() => setShowAddressModal(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>주소 검색</Text>
            <TouchableOpacity
              style={styles.modalCloseButton}
              onPress={() => setShowAddressModal(false)}
            >
              <Text style={styles.modalCloseText}>✕</Text>
            </TouchableOpacity>
          </View>
          <WebView
            key="daum-postcode"
            source={{ html: daumPostcodeHtml, baseUrl: 'https://postcode.map.daum.net' }}
            onMessage={handleAddressMessage}
            style={styles.webView}
            javaScriptEnabled={true}
            domStorageEnabled={true}
            originWhitelist={['*']}
            onError={(e) => console.log('WebView error:', e.nativeEvent)}
            onHttpError={(e) => console.log('WebView HTTP error:', e.nativeEvent)}
            startInLoadingState={true}
            cacheEnabled={false}
          />
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingTop: Platform.OS === 'ios' ? 50 : 20,
    paddingBottom: 16,
    backgroundColor: colors.primary,
  },
  backButton: {
    padding: 8,
  },
  backButtonText: {
    color: '#ffffff',
    fontSize: 16,
  },
  headerTitle: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: '600',
  },
  headerRight: {
    width: 60,
  },
  content: {
    flex: 1,
    padding: 16,
  },
  formGroup: {
    marginBottom: 20,
  },
  label: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.text,
    marginBottom: 8,
  },
  input: {
    backgroundColor: colors.card,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    color: colors.text,
  },
  inputError: {
    borderColor: colors.red,
  },
  inputRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  inputWithUnit: {
    flex: 1,
    marginRight: 8,
  },
  unitText: {
    fontSize: 16,
    color: colors.textMuted,
    width: 40,
  },
  helpText: {
    fontSize: 12,
    color: colors.textMuted,
    marginTop: 4,
  },
  errorText: {
    fontSize: 12,
    color: colors.red,
    marginTop: 4,
  },
  segmentedControl: {
    flexDirection: 'row',
    backgroundColor: colors.card,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: colors.border,
    overflow: 'hidden',
  },
  segmentButton: {
    flex: 1,
    paddingVertical: 12,
    alignItems: 'center',
    borderRightWidth: 1,
    borderRightColor: colors.border,
  },
  segmentButtonActive: {
    backgroundColor: colors.primary,
  },
  segmentButtonText: {
    fontSize: 14,
    color: colors.text,
  },
  segmentButtonTextActive: {
    color: '#ffffff',
    fontWeight: '600',
  },
  dateRow: {
    flexDirection: 'row',
    gap: 16,
  },
  dateInput: {
    flex: 1,
  },
  dateLabel: {
    fontSize: 12,
    color: colors.textMuted,
    marginBottom: 4,
  },
  pickerContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.card,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 8,
  },
  pickerButton: {
    padding: 12,
    backgroundColor: colors.background,
  },
  pickerButtonText: {
    fontSize: 18,
    color: colors.primary,
    fontWeight: '600',
  },
  pickerValue: {
    flex: 1,
    textAlign: 'center',
    fontSize: 16,
    color: colors.text,
  },
  efficiencyCard: {
    backgroundColor: colors.card,
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: colors.border,
  },
  efficiencyLabel: {
    fontSize: 14,
    color: colors.textMuted,
    marginBottom: 4,
  },
  efficiencyRow: {
    flexDirection: 'row',
    alignItems: 'baseline',
    marginBottom: 8,
  },
  efficiencyValue: {
    fontSize: 32,
    fontWeight: '700',
  },
  efficiencyYear: {
    fontSize: 14,
    color: colors.textMuted,
    marginLeft: 8,
  },
  efficiencyHelp: {
    fontSize: 12,
    color: colors.textMuted,
  },
  radioGroup: {
    gap: 12,
  },
  radioItem: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.card,
    borderRadius: 8,
    padding: 12,
    borderWidth: 1,
    borderColor: colors.border,
  },
  radio: {
    width: 20,
    height: 20,
    borderRadius: 10,
    borderWidth: 2,
    borderColor: colors.border,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  radioActive: {
    borderColor: colors.primary,
  },
  radioDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: colors.primary,
  },
  radioContent: {
    flex: 1,
  },
  radioLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.text,
  },
  radioDesc: {
    fontSize: 12,
    color: colors.textMuted,
  },
  directionRow: {
    flexDirection: 'row',
    gap: 8,
  },
  directionButton: {
    flex: 1,
    paddingVertical: 10,
    paddingHorizontal: 12,
    backgroundColor: colors.card,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: colors.border,
    alignItems: 'center',
  },
  directionButtonActive: {
    backgroundColor: colors.primary,
    borderColor: colors.primary,
  },
  directionButtonText: {
    fontSize: 14,
    color: colors.text,
  },
  directionButtonTextActive: {
    color: '#ffffff',
    fontWeight: '600',
  },
  estimatesCard: {
    backgroundColor: '#f0fdf4',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#bbf7d0',
  },
  estimatesTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.text,
    marginBottom: 12,
  },
  estimatesGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  estimateItem: {
    width: '50%',
    marginBottom: 12,
  },
  estimateLabel: {
    fontSize: 12,
    color: colors.textMuted,
    marginBottom: 2,
  },
  estimateValue: {
    fontSize: 18,
    fontWeight: '600',
    color: colors.text,
  },
  estimatesHelp: {
    fontSize: 11,
    color: colors.textMuted,
    fontStyle: 'italic',
  },
  buttonGroup: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 8,
  },
  cancelButton: {
    flex: 1,
    paddingVertical: 14,
    backgroundColor: colors.card,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: colors.border,
    alignItems: 'center',
  },
  cancelButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.textMuted,
  },
  submitButton: {
    flex: 1,
    paddingVertical: 14,
    backgroundColor: colors.primary,
    borderRadius: 8,
    alignItems: 'center',
  },
  submitButtonDisabled: {
    opacity: 0.6,
  },
  submitButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#ffffff',
  },
  // Address search styles
  addressRow: {
    flexDirection: 'row',
    gap: 8,
  },
  addressInput: {
    flex: 1,
  },
  addressSearchButton: {
    backgroundColor: colors.primary,
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 8,
    justifyContent: 'center',
    alignItems: 'center',
  },
  addressSearchButtonText: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: '600',
  },
  clearAddressText: {
    fontSize: 12,
    color: colors.red,
    marginTop: 6,
  },
  // Modal styles
  modalContainer: {
    flex: 1,
    backgroundColor: colors.background,
  },
  modalHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingTop: Platform.OS === 'ios' ? 50 : 20,
    paddingBottom: 16,
    backgroundColor: colors.primary,
  },
  modalTitle: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: '600',
  },
  modalCloseButton: {
    padding: 8,
  },
  modalCloseText: {
    color: '#ffffff',
    fontSize: 20,
    fontWeight: '600',
  },
  webView: {
    flex: 1,
  },
});
