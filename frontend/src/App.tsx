import { useEffect, useMemo, useState, type ReactNode } from 'react'
import {
  Activity,
  ArrowRight,
  BarChart3,
  Database,
  FlaskConical,
  Gauge,
  Layers3,
  LoaderCircle,
  Scale,
  ShieldAlert,
  Sparkles,
  Target,
  Waves,
} from 'lucide-react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  ErrorBar,
  Line,
  LineChart,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

type View = 'overview' | 'predict' | 'model'

type ModelCard = {
  model_name: string
  algorithm: string
  dataset_source: string
  intended_use: string
  pipeline_steps: string[]
  hyperparameters: Record<string, string | number>
  performance_metrics: Record<string, number>
  age_group_mae: Record<string, number>
  limitations: string[]
  feature_statistics: Record<string, { desc: string; unit: string; min: number; max: number; mean: number; std: number }>
}

type EvaluationReports = {
  sources: Record<string, string>
  dataAudit: {
    rawSamples: number
    removedSamples: number
    cleanedSamples: number
    trainingSamples: number
    testSamples: number
  }
  modelResults: Array<{ model: string; mae: number; rmse: number; r2: number }>
  cvAblation: Array<{ preprocessing: string; bestK: number; bestWeights: string; cvMae: number; cvStd: number }>
  cvResults: Array<{ scaling: string; withScaling: boolean; weights: string; k: number; mae: number; maeStd: number }>
  ageGroupMae: Array<{ group: string; mae: number }>
  metadata: {
    totalTrainingSamples: number
    totalTestSamples: number
    hyperparameters: Record<string, string | number>
    pipelineSteps: string[]
    intendedUse: string
    limitations: string[]
    datasetSource: string
    featureStatistics: Record<string, {
      description: string
      min: number
      max: number
      mean: number
      physicalUnit: string
    }>
  }
}

type FeatureRangeItem = {
  unit: string
  description: string
  min: number
  max: number
  mean: number
  std: number
  physical_unit: string
  physical_min: number
  physical_max: number
  physical_mean: number
}

type FeatureRanges = {
  numeric_ranges: Record<string, FeatureRangeItem>
  valid_sex_values: string[]
  total_training_samples: number
  scale_factor: number
  scaling_explanation: string
}

type PredictionResponse = {
  predicted_rings: number
  estimated_age_years: number
  age_group: 'Young' | 'Middle' | 'Old'
  reference_interval: {
    min_rings: number
    max_rings: number
    median_rings: number
    min_age_years: number
    max_age_years: number
  }
  nearest_neighbors: Array<{
    distance: number
    rings: number
    estimated_age: number
  }>
  is_out_of_distribution: boolean
  warnings: string[]
}

type FormState = {
  Sex: 'M' | 'F' | 'I'
  Length: string
  Diameter: string
  Height: string
  Whole_weight: string
  Shucked_weight: string
  Viscera_weight: string
  Shell_weight: string
}

type MeasurementField = Exclude<keyof FormState, 'Sex'>
type InputScale = 'uci' | 'physical'

type MetricCardProps = {
  label: string
  value: string
  hint: string
  icon: ReactNode
  tone?: 'primary' | 'success' | 'neutral'
}

const defaultForm: FormState = {
  Sex: 'M',
  Length: '0.455',
  Diameter: '0.365',
  Height: '0.095',
  Whole_weight: '0.514',
  Shucked_weight: '0.2245',
  Viscera_weight: '0.101',
  Shell_weight: '0.15',
}

const API_BASE = '/api'

const samplePresets: Record<'standard' | 'physical' | 'ood', FormState> = {
  standard: { ...defaultForm },
  physical: {
    Sex: 'M',
    Length: '91',
    Diameter: '73',
    Height: '19',
    Whole_weight: '102.8',
    Shucked_weight: '44.9',
    Viscera_weight: '20.2',
    Shell_weight: '30',
  },
  ood: {
    Sex: 'F',
    Length: '0.95',
    Diameter: '0.75',
    Height: '0.35',
    Whole_weight: '3.5',
    Shucked_weight: '1.6',
    Viscera_weight: '0.85',
    Shell_weight: '1.1',
  },
}

const measurementFields: Array<{
  key: MeasurementField
  label: string
  physicalUnit: 'mm' | 'g'
}> = [
  { key: 'Length', label: 'Length · Chiều dài vỏ', physicalUnit: 'mm' },
  { key: 'Diameter', label: 'Diameter · Đường kính vỏ', physicalUnit: 'mm' },
  { key: 'Height', label: 'Height · Chiều cao vỏ', physicalUnit: 'mm' },
  { key: 'Whole_weight', label: 'Whole weight · Khối lượng toàn bộ', physicalUnit: 'g' },
  { key: 'Shucked_weight', label: 'Shucked weight · Khối lượng thịt', physicalUnit: 'g' },
  { key: 'Viscera_weight', label: 'Viscera weight · Khối lượng nội tạng', physicalUnit: 'g' },
  { key: 'Shell_weight', label: 'Shell weight · Khối lượng vỏ khô', physicalUnit: 'g' },
]

const featureDescriptions: Record<string, string> = {
  Sex: 'Giới tính: M (đực), F (cái), I (non/chưa phân hóa).',
  Length: 'Chiều dài lớn nhất của vỏ, song song trục dài.',
  Diameter: 'Đường kính lớn nhất vuông góc với chiều dài.',
  Height: 'Chiều cao thẳng đứng gồm phần thịt bên trong vỏ.',
  Whole_weight: 'Khối lượng nguyên con: vỏ, thịt và nội tạng.',
  Shucked_weight: 'Khối lượng phần thịt sau khi tách vỏ.',
  Viscera_weight: 'Khối lượng nội tạng sau khi mổ.',
  Shell_weight: 'Khối lượng vỏ sau khi rửa và sấy khô.',
}

const safeLimitations = [
  'Tuổi được ước lượng từ Rings + 1.5 năm theo quy ước dữ liệu; đây không phải phép đo tuổi sinh học trực tiếp.',
  'Bảng phân tích MAE theo nhóm tuổi không kèm số mẫu mỗi nhóm; không thể kết luận nguyên nhân chênh lệch sai số từ artifact này.',
  'Metric mô tả tập test của thí nghiệm hiện có; không bảo đảm hiệu năng tương tự trên dữ liệu ngoài miền.',
]

const cvLineColors = {
  scaled_uniform: '#1f7ae0',
  scaled_distance: '#12a561',
  unscaled_uniform: '#7b61c9',
  unscaled_distance: '#e59a27',
}

const formatNumber = (value: number | undefined, digits = 3) => {
  if (value === undefined || Number.isNaN(value)) return '—'
  return Number(value).toFixed(digits)
}

function formatApiError(detail: unknown): string {
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail
      .map((issue) => {
        if (typeof issue === 'string') return issue
        if (issue && typeof issue === 'object' && 'msg' in issue) {
          const location = 'loc' in issue && Array.isArray(issue.loc)
            ? issue.loc.filter((part: unknown) => part !== 'body').join(' → ')
            : ''
          return `${location ? `${location}: ` : ''}${String(issue.msg)}`
        }
        return JSON.stringify(issue)
      })
      .join('; ')
  }
  return 'Không thể thực hiện yêu cầu với dữ liệu hiện tại.'
}

function formatOodWarning(warning: string): { message: string; usesNormalizedValues: boolean } {
  const message = warning
    .replace(/(\d)\s*(?:mm|g)\b/gi, '$1')
    .replace(/\s+\./g, '.')

  return {
    message,
    usesNormalizedValues: message !== warning,
  }
}

function MetricCard({ label, value, hint, icon, tone = 'neutral' }: MetricCardProps) {
  return (
    <div className={`stat-card ${tone}`}>
      <div className="stat-header">
        <span className="stat-label">{label}</span>
        <span className="stat-icon">{icon}</span>
      </div>
      <div className="stat-value">{value}</div>
      <div className="stat-hint">{hint}</div>
    </div>
  )
}

function App() {
  const [activeView, setActiveView] = useState<View>('overview')
  const [modelCard, setModelCard] = useState<ModelCard | null>(null)
  const [featureRanges, setFeatureRanges] = useState<FeatureRanges | null>(null)
  const [evaluationReports, setEvaluationReports] = useState<EvaluationReports | null>(null)
  const [form, setForm] = useState<FormState>(defaultForm)
  const [inputScale, setInputScale] = useState<InputScale>('uci')
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null)
  const [showAllNeighbors, setShowAllNeighbors] = useState(false)
  const [loading, setLoading] = useState(true)
  const [reportsLoading, setReportsLoading] = useState(true)
  const [apiAvailable, setApiAvailable] = useState<boolean | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [reportsError, setReportsError] = useState<string | null>(null)

  useEffect(() => {
    const loadDashboard = async () => {
      try {
        setLoading(true)
        const [modelResponse, featureResponse] = await Promise.all([
          fetch(`${API_BASE}/model-card`),
          fetch(`${API_BASE}/feature-ranges`),
        ])

        if (!modelResponse.ok || !featureResponse.ok) {
          throw new Error('Không thể tải dữ liệu từ API. Hãy khởi động backend trước.')
        }

        const modelData = (await modelResponse.json()) as ModelCard
        const featureData = (await featureResponse.json()) as FeatureRanges

        setModelCard(modelData)
        setFeatureRanges(featureData)
        setError(null)
        setApiAvailable(true)
      } catch (loadError) {
        setError(
          loadError instanceof TypeError
            ? 'Không thể kết nối backend API. Hãy khởi động FastAPI để xem metric và cấu hình runtime.'
            : loadError instanceof Error
              ? loadError.message
              : 'Lỗi không xác định khi tải dữ liệu.',
        )
        setApiAvailable(false)
      } finally {
        setLoading(false)
      }
    }

    const loadReports = async () => {
      try {
        const response = await fetch('/model-reports/evaluation.json')
        if (!response.ok) {
          throw new Error(`Không thể tải báo cáo tĩnh (${response.status}). Chạy lại npm run predev để tái tạo dữ liệu.`)
        }
        setEvaluationReports((await response.json()) as EvaluationReports)
      } catch (loadError) {
        setReportsError(loadError instanceof Error ? loadError.message : 'Không thể tải artifact thực nghiệm.')
      } finally {
        setReportsLoading(false)
      }
    }

    void loadDashboard()
    void loadReports()
  }, [])

  const cvChartData = useMemo(() => {
    if (!evaluationReports) return []
    const grouped = new Map<number, Record<string, number | null>>()

    evaluationReports.cvResults.forEach((row) => {
      const point = grouped.get(row.k) ?? { k: row.k }
      const series = `${row.withScaling ? 'scaled' : 'unscaled'}_${row.weights}`
      point[series] = row.mae
      grouped.set(row.k, point)
    })

    return [...grouped.values()].sort((first, second) => Number(first.k) - Number(second.k))
  }, [evaluationReports])

  const selectedCvResult = useMemo(() => {
    if (!evaluationReports) return null
    const hyperparameters = modelCard?.hyperparameters ?? evaluationReports.metadata.hyperparameters
    const usesScaling = String(hyperparameters.scaling ?? '').toLowerCase().includes('scaler')
    return evaluationReports.cvResults.find(
      (row) =>
        row.k === Number(hyperparameters.n_neighbors) &&
        row.weights === String(hyperparameters.weights) &&
        row.withScaling === usesScaling,
    ) ?? null
  }, [evaluationReports, modelCard])

  const apiMatchesReport = useMemo(() => {
    if (!modelCard || !evaluationReports) return false
    const runtimeHyperparameters = modelCard.hyperparameters
    const artifactHyperparameters = evaluationReports.metadata.hyperparameters
    const standardScalerResult = evaluationReports.modelResults.find(
      (row) => row.model === 'KNN - StandardScaler',
    )
    const metrics = modelCard.performance_metrics
    const close = (runtime: number | undefined, artifact: number | undefined) =>
      runtime !== undefined && artifact !== undefined && Math.abs(runtime - artifact) < 0.00011

    return (
      Number(runtimeHyperparameters.n_neighbors) === Number(artifactHyperparameters.n_neighbors) &&
      String(runtimeHyperparameters.weights) === String(artifactHyperparameters.weights) &&
      String(runtimeHyperparameters.scaling) === String(artifactHyperparameters.scaling) &&
      standardScalerResult !== undefined &&
      close(metrics.Test_MAE, standardScalerResult.mae) &&
      close(metrics.Test_RMSE, standardScalerResult.rmse) &&
      close(metrics.Test_R2, standardScalerResult.r2)
    )
  }, [evaluationReports, modelCard])

  const featureEntries = useMemo(
    () => {
      if (featureRanges) return Object.entries(featureRanges.numeric_ranges)
      if (!evaluationReports) return []
      return Object.entries(evaluationReports.metadata.featureStatistics).map(([feature, stats]) => [
        feature,
        {
          unit: stats.physicalUnit,
          description: stats.description,
          min: stats.min / 200,
          max: stats.max / 200,
          mean: stats.mean / 200,
          std: 0,
          physical_unit: stats.physicalUnit,
          physical_min: stats.min,
          physical_max: stats.max,
          physical_mean: stats.mean,
        },
      ] as [string, FeatureRangeItem])
    },
    [evaluationReports, featureRanges],
  )

  const getObservedRange = (feature: MeasurementField) => {
    const apiRange = featureRanges?.numeric_ranges[feature]
    if (apiRange) {
      return {
        min: apiRange.physical_min,
        max: apiRange.physical_max,
        unit: apiRange.physical_unit,
      }
    }

    const reportRange = evaluationReports?.metadata.featureStatistics[feature]
    return reportRange
      ? { min: reportRange.min, max: reportRange.max, unit: reportRange.physicalUnit }
      : null
  }

  const handleFieldChange = (field: keyof FormState, rawValue: string) => {
    setForm((previous) => ({ ...previous, [field]: rawValue }))
  }

  const changeInputScale = (nextScale: InputScale) => {
    if (nextScale === inputScale) return
    setForm((previous) => {
      const converted = { ...previous }
      measurementFields.forEach(({ key }) => {
        const current = previous[key]
        const numeric = Number(current)
        if (current.trim() && Number.isFinite(numeric)) {
          converted[key] = String(Number((numeric * (nextScale === 'physical' ? 200 : 1 / 200)).toPrecision(12)))
        }
      })
      return converted
    })
    setInputScale(nextScale)
    setError(null)
    setPrediction(null)
  }

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setSubmitting(true)
    setError(null)

    try {
      const numericValues: Partial<Record<MeasurementField, number>> = {}
      for (const { key, label } of measurementFields) {
        const rawValue = form[key].trim()
        if (!rawValue) throw new Error(`${label}: vui lòng nhập giá trị.`)
        const value = Number(rawValue)
        if (!Number.isFinite(value)) throw new Error(`${label}: cần nhập một số hợp lệ.`)
        if (value <= 0) throw new Error(`${label}: giá trị phải lớn hơn 0.`)
        numericValues[key] = inputScale === 'physical' ? Number((value / 200).toPrecision(12)) : value
      }

      const payload = {
        Sex: form.Sex,
        ...numericValues,
      }
      const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })

      if (!response.ok) {
        const payload = await response.json().catch(() => null)
        throw new Error(formatApiError(payload?.detail))
      }

      const result = (await response.json()) as PredictionResponse
      setPrediction(result)
      setShowAllNeighbors(false)
      setActiveView('predict')
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : 'Lỗi khi thực hiện dự đoán.')
    } finally {
      setSubmitting(false)
    }
  }

  const loadPreset = (key: 'standard' | 'physical' | 'ood') => {
    setForm(samplePresets[key])
    setInputScale(key === 'physical' ? 'physical' : 'uci')
    setPrediction(null)
    setShowAllNeighbors(false)
    setError(null)
  }

  const resetPrediction = () => {
    setForm(defaultForm)
    setInputScale('uci')
    setPrediction(null)
    setShowAllNeighbors(false)
    setError(null)
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-wrap">
          <div className="brand-mark">
            <Waves size={18} />
          </div>
          <div>
            <div className="brand-title">Abalone AI</div>
            <div className="brand-subtitle">Marine Life Research Dashboard</div>
          </div>
        </div>

        <nav className="nav-tabs" aria-label="Navigation">
          {[
            { key: 'overview', label: 'Tổng quan' },
            { key: 'predict', label: 'Dự đoán' },
            { key: 'model', label: 'Mô hình' },
          ].map((tab) => (
            <button
              key={tab.key}
              type="button"
              className={`tab-button ${activeView === tab.key ? 'active' : ''}`}
              onClick={() => setActiveView(tab.key as View)}
            >
              {tab.label}
            </button>
          ))}
        </nav>

        <div className={`status-pill ${apiAvailable === false && !loading ? 'offline' : ''}`}>
          <span className="pulse" />
          {loading ? 'Đang đồng bộ' : apiAvailable ? 'API sẵn sàng' : 'API ngoại tuyến'}
        </div>
      </header>

      {error ? <div className="banner error-banner">{error}</div> : null}

      {activeView === 'overview' ? (
        <main className="page-section overview-page">
          <section className="hero-panel">
            <div className="hero-copy">
              <div className="eyebrow">Project 06 · K-NN Regression</div>
              <h1>Ước lượng tuổi bào ngư bằng K-NN Regression</h1>
              <p>
                Ứng dụng học máy hỗ trợ ước lượng tuổi bào ngư từ các phép đo hình thái và khối
                lượng, đồng thời cung cấp thông tin về khả năng ngoại lệ và khoảng tham chiếu gần
                nhất của từng mẫu.
              </p>
              <div className="action-row">
                <button type="button" className="primary-button" onClick={() => setActiveView('predict')}>
                  Bắt đầu dự đoán
                  <ArrowRight size={18} />
                </button>
                <button type="button" className="secondary-button" onClick={() => setActiveView('model')}>
                  Khám phá mô hình
                </button>
              </div>
            </div>

            <div className="hero-visual">
              <div className="shell-card glass-card">
                <div className="shell-illustration">
                  <div className="shell-shell" />
                </div>
                <div className="illustration-caption">Abalone shell morphology</div>
              </div>
            </div>
          </section>

          <section className="metrics-grid">
            <MetricCard
              label="Dataset"
              value={evaluationReports ? evaluationReports.dataAudit.rawSamples.toLocaleString('vi-VN') : '—'}
              hint="Mẫu raw · UCI Abalone"
              icon={<Database size={18} />}
              tone="primary"
            />
            <MetricCard
              label="Train / Test"
              value={evaluationReports ? `${evaluationReports.dataAudit.trainingSamples.toLocaleString('vi-VN')} / ${evaluationReports.dataAudit.testSamples.toLocaleString('vi-VN')}` : '—'}
              hint={evaluationReports ? `${evaluationReports.dataAudit.cleanedSamples.toLocaleString('vi-VN')} mẫu sau audit · ${evaluationReports.dataAudit.removedSamples} dòng loại` : 'Số lượng theo metadata'}
              icon={<Layers3 size={18} />}
              tone="success"
            />
            <MetricCard
              label="Features"
              value="8"
              hint="Mô hình dự đoán vòng vỏ"
              icon={<Target size={18} />}
            />
            <MetricCard
              label="MAE test"
              value={modelCard ? formatNumber(modelCard.performance_metrics.Test_MAE, 3) : '—'}
              hint="Rings · sai số tuyệt đối trung bình"
              icon={<Gauge size={18} />}
            />
          </section>

          <section className="content-grid">
            <article className="panel section-card">
              <div className="section-header">
                <FlaskConical size={18} />
                <h2>Vấn đề nghiên cứu</h2>
              </div>
              <p>
                Xác định tuổi bào ngư bằng cách đếm vòng vỏ trên vỏ là quá trình tốn thời gian và
                phụ thuộc nhiều vào kỹ thuật vi thể. Mô hình này hỗ trợ xác định độ tuổi sơ bộ từ các
                phép đo hình thái và khối lượng, nhưng không thay thế quy trình sinh học chuẩn.
              </p>
            </article>

            <article className="panel section-card">
              <div className="section-header">
                <Sparkles size={18} />
                <h2>Quy trình dữ liệu và đánh giá</h2>
              </div>
              {evaluationReports ? (
                <div className="pipeline-diagram" aria-label="Luồng dữ liệu, train, cross-validation, test và serving">
                  <div className="flow-node source-node">
                    <strong>UCI raw</strong>
                    <span>{evaluationReports.dataAudit.rawSamples.toLocaleString('vi-VN')} dòng</span>
                  </div>
                  <div className="flow-arrow" aria-hidden="true">→</div>
                  <div className="flow-node">
                    <strong>Làm sạch</strong>
                    <span>Loại {evaluationReports.dataAudit.removedSamples} dòng theo anomaly audit</span>
                  </div>
                  <div className="flow-arrow" aria-hidden="true">→</div>
                  <div className="flow-node">
                    <strong>Chia Train / Test</strong>
                    <span>{evaluationReports.dataAudit.trainingSamples.toLocaleString('vi-VN')} / {evaluationReports.dataAudit.testSamples.toLocaleString('vi-VN')} mẫu</span>
                  </div>
                  <div className="flow-branches">
                    <article className="flow-lane train-lane">
                      <div className="lane-label">TRAIN · chỉ dùng tập huấn luyện</div>
                      <div className="flow-lane-steps">
                        <div className="flow-node">
                          <strong>5-fold CV</strong>
                          <span>So sánh K và weights trên Train</span>
                        </div>
                        <span className="flow-arrow" aria-hidden="true">→</span>
                        <div className="flow-node">
                          <strong>Chọn cấu hình</strong>
                          <span>Mã pipeline chọn trong nhánh StandardScaler</span>
                        </div>
                        <span className="flow-arrow" aria-hidden="true">→</span>
                        <div className="flow-node">
                          <strong>Fit pipeline cuối</strong>
                          <span>Preprocessor + KNN fit trên toàn bộ Train</span>
                        </div>
                      </div>
                    </article>
                    <article className="flow-lane test-lane">
                      <div className="lane-label">TEST · giữ độc lập, không chọn cấu hình</div>
                      <p>Đánh giá cuối cùng và phân tích residual/MAE nhóm tuổi trên tập test.</p>
                    </article>
                    <article className="flow-lane serving-lane">
                      <div className="lane-label">SERVING</div>
                      <p>Pipeline đã lưu được nạp bởi FastAPI; frontend gửi phép đo đến Prediction API.</p>
                    </article>
                  </div>
                  <p className="source-note">
                    Nguồn số lượng: {evaluationReports.sources.rawDataset}, {evaluationReports.sources.removedAnomalies} và {evaluationReports.sources.metadata}.
                  </p>
                </div>
              ) : (
                <div className="empty-state">Đang tải dữ liệu đối chiếu pipeline…</div>
              )}
            </article>
          </section>

          <section className="panel section-card">
            <div className="section-header">
              <Activity size={18} />
              <h2>8 đặc trưng chính</h2>
            </div>

            <div className="feature-grid">
              {featureEntries.length > 0 ? (
                [
                  ['Sex', null] as const,
                  ...featureEntries.map(([feature, stats]) => [feature, stats] as const),
                ].map(([feature, stats]) => (
                  <div key={feature} className="feature-card">
                    <div className="feature-name">{feature}</div>
                    <div className="feature-meta">{featureDescriptions[feature]}</div>
                    {stats ? (
                      <div className="feature-range">
                        Quan sát Train: {formatNumber(stats.physical_min, 2)}–{formatNumber(stats.physical_max, 2)} {stats.physical_unit}
                      </div>
                    ) : (
                      <div className="feature-range">
                        Giá trị: {featureRanges?.valid_sex_values.join(' · ') ?? 'M · F · I'}
                      </div>
                    )}
                  </div>
                ))
              ) : (
                <div className="empty-state">Đang tải thông tin đặc trưng…</div>
              )}
            </div>
          </section>
        </main>
      ) : null}

      {activeView === 'predict' ? (
        <main className="page-section predict-page">
          <section className="panel form-panel">
            <div className="section-header">
              <Target size={18} />
              <h2>Nhập dữ liệu bào ngư</h2>
            </div>

            <div className="preset-row">
              <button type="button" className="preset-button" onClick={() => loadPreset('standard')}>
                Ví dụ hợp lệ · thang UCI
              </button>
              <button type="button" className="preset-button" onClick={() => loadPreset('physical')}>
                Ví dụ vật lý · mm/g
              </button>
              <button type="button" className="preset-button" onClick={() => loadPreset('ood')}>
                Ví dụ ngoài miền · thang UCI
              </button>
            </div>
            <p className="preset-help">
              Các nút chỉ nạp ví dụ đã kiểm thử/có trong API schema; hãy nhấn “Dự đoán tuổi” để gửi mẫu.
              Ví dụ ngoài miền vẫn hợp lệ về schema nhưng vượt miền quan sát của một số đặc trưng.
            </p>

            <form className="prediction-form" onSubmit={handleSubmit}>
              <fieldset className="unit-selector">
                <legend>Đơn vị nhập cho 7 phép đo liên tục</legend>
                <label className="radio-option">
                  <input
                    type="radio"
                    name="input-scale"
                    checked={inputScale === 'uci'}
                    onChange={() => changeInputScale('uci')}
                  />
                  Thang dữ liệu UCI (đã chia 200)
                </label>
                <label className="radio-option">
                  <input
                    type="radio"
                    name="input-scale"
                    checked={inputScale === 'physical'}
                    onChange={() => changeInputScale('physical')}
                  />
                  Đơn vị vật lý (mm / g)
                </label>
                <p>
                  {inputScale === 'physical'
                    ? 'Frontend tự chia từng giá trị cho 200 trước khi gửi API.'
                    : 'Giá trị nhập được gửi nguyên trạng theo thang UCI.'}
                  {' '}Dùng dấu chấm cho phần thập phân.
                </p>
              </fieldset>
              <div className="form-row">
                <label>
                  <span>Sex · Giới tính</span>
                  <select value={form.Sex} onChange={(event) => handleFieldChange('Sex', event.target.value)}>
                    <option value="M">M · Đực</option>
                    <option value="F">F · Cái</option>
                    <option value="I">I · Non / chưa phân hóa</option>
                  </select>
                </label>
              </div>

              <div className="form-grid">
                {measurementFields.map(({ key, label, physicalUnit }) => {
                  const observedRange = getObservedRange(key)
                  return (
                  <label key={key} className="measurement-label">
                    <span>{label} <em>({inputScale === 'physical' ? physicalUnit : 'thang UCI'})</em></span>
                    <input
                      type="text"
                      inputMode="decimal"
                      autoComplete="off"
                      value={form[key]}
                      aria-describedby={`${key}-measurement-help`}
                      onChange={(event) => handleFieldChange(key, event.target.value)}
                    />
                    <small id={`${key}-measurement-help`}>
                      {observedRange
                        ? `Train quan sát: ${formatNumber(inputScale === 'physical' ? observedRange.min : observedRange.min / 200, 4)}–${formatNumber(inputScale === 'physical' ? observedRange.max : observedRange.max / 200, 4)} ${inputScale === 'physical' ? observedRange.unit : '(đã chia 200)'}. Nhập số dương; ngoài miền vẫn có thể gửi để kiểm tra OOD.`
                        : 'Nhập số dương; ngoài miền vẫn có thể gửi để kiểm tra OOD.'}
                    </small>
                  </label>
                  )
                })}
              </div>

              <div className="form-actions">
                <button type="submit" className="primary-button submit-button" disabled={submitting}>
                  {submitting ? (
                    <>
                      <LoaderCircle size={18} className="spinner" />
                      Đang dự đoán...
                    </>
                  ) : (
                    'Dự đoán tuổi'
                  )}
                </button>
                <button type="button" className="preset-button reset-button" onClick={resetPrediction}>
                  Đặt lại
                </button>
              </div>
            </form>
          </section>

          <section className="panel result-panel">
            <div className="section-header">
              <BarChart3 size={18} />
              <h2>Kết quả dự đoán</h2>
            </div>

            {prediction ? (
              <>
                <div className="result-badges">
                  <span className={`badge ${prediction.age_group.toLowerCase()}`}>
                    Nhóm theo Rings dự đoán: {prediction.age_group}
                  </span>
                  {prediction.is_out_of_distribution ? (
                    <span className="badge warn"><ShieldAlert size={14} /> Ngoài miền dữ liệu</span>
                  ) : (
                    <span className="badge ok">Trong miền dữ liệu</span>
                  )}
                </div>

                <div className="result-stats">
                  <div>
                    <span className="result-label">Rings dự đoán</span>
                    <strong>{formatNumber(prediction.predicted_rings, 2)}</strong>
                  </div>
                  <div>
                    <span className="result-label">Tuổi ước tính</span>
                    <strong>{formatNumber(prediction.estimated_age_years, 2)} năm</strong>
                  </div>
                </div>

                <div className="interval-box">
                  <div className="interval-title">Khoảng giá trị trong nhóm láng giềng</div>
                  <div className="interval-grid">
                    <div>
                      <span>Min</span>
                      <strong>{formatNumber(prediction.reference_interval.min_rings, 2)}</strong>
                    </div>
                    <div>
                      <span>Median</span>
                      <strong>{formatNumber(prediction.reference_interval.median_rings, 2)}</strong>
                    </div>
                    <div>
                      <span>Max</span>
                      <strong>{formatNumber(prediction.reference_interval.max_rings, 2)}</strong>
                    </div>
                  </div>
                  <p className="result-explainer">
                    Min, median và max Rings của nearest neighbors do API trả về. Đây là khoảng tham chiếu,
                    không phải khoảng tin cậy 95%.
                  </p>
                </div>

                {prediction.is_out_of_distribution || prediction.warnings.length > 0 ? (
                  <details className="ood-summary" open={prediction.is_out_of_distribution}>
                    <summary>
                      <ShieldAlert size={16} />
                      Cảnh báo dữ liệu ngoài miền ({prediction.warnings.length})
                    </summary>
                    {prediction.warnings.length > 0 ? (
                      <>
                        <ul>
                          {prediction.warnings.map((warning) => {
                            const formatted = formatOodWarning(warning)
                            return <li key={warning}>{formatted.message}</li>
                          })}
                        </ul>
                        {prediction.warnings.some((warning) => formatOodWarning(warning).usesNormalizedValues) ? (
                          <p className="ood-unit-note">
                            Các giá trị trong cảnh báo được API tính trên payload thang UCI đã chia 200.
                            Đơn vị vật lý tương ứng được quy đổi bằng cách nhân 200 (mm cho kích thước, g cho khối lượng).
                          </p>
                        ) : null}
                      </>
                    ) : (
                      <p>API đánh dấu mẫu ngoài miền nhưng không trả về nội dung cảnh báo chi tiết.</p>
                    )}
                  </details>
                ) : null}

                <div className="neighbors-wrap">
                  <h3>Nearest neighbors · {prediction.nearest_neighbors.length} mẫu trả về</h3>
                  <p className="chart-caption">
                    Đây là các mẫu huấn luyện gần nhất theo khoảng cách của pipeline, không phải mẫu test.
                  </p>
                  <div className="neighbor-list">
                    {prediction.nearest_neighbors
                      .slice(0, showAllNeighbors ? prediction.nearest_neighbors.length : 5)
                      .map((neighbor, index) => (
                      <div className="neighbor-item" key={`${neighbor.rings}-${index}`}>
                        <div className="neighbor-head">
                          <span>NN {index + 1}</span>
                          <strong>{formatNumber(neighbor.rings, 1)} rings</strong>
                        </div>
                        <div className="neighbor-meta">
                          Khoảng cách: {formatNumber(neighbor.distance, 4)} · Tuổi: {formatNumber(neighbor.estimated_age, 2)} năm
                        </div>
                      </div>
                    ))}
                  </div>
                  {prediction.nearest_neighbors.length > 5 ? (
                    <button
                      type="button"
                      className="text-button"
                      onClick={() => setShowAllNeighbors((shown) => !shown)}
                    >
                      {showAllNeighbors ? 'Thu gọn còn 5 mẫu' : `Xem đầy đủ ${prediction.nearest_neighbors.length} mẫu`}
                    </button>
                  ) : null}
                </div>
              </>
            ) : (
              <div className="empty-result">
                Chưa có kết quả dự đoán. Hãy nhập các thông số và gửi yêu cầu để xem ước lượng tuổi.
              </div>
            )}
          </section>
        </main>
      ) : null}

      {activeView === 'model' ? (
        <main className="page-section model-page">
          <section className="panel section-card experiment-card">
            <div className="section-header">
              <Gauge size={18} />
              <h2>Hiệu năng mô hình đang được API phục vụ</h2>
            </div>
            <div className="model-kpis">
              <MetricCard
                label="MAE · Test"
                value={modelCard ? formatNumber(modelCard.performance_metrics.Test_MAE, 4) : '—'}
                hint="Rings · sai số tuyệt đối trung bình"
                icon={<Target size={18} />}
                tone="primary"
              />
              <MetricCard
                label="RMSE · Test"
                value={modelCard ? formatNumber(modelCard.performance_metrics.Test_RMSE, 4) : '—'}
                hint="Rings · nhạy hơn với sai số lớn"
                icon={<Activity size={18} />}
                tone="success"
              />
              <MetricCard
                label="R² · Test"
                value={modelCard ? formatNumber(modelCard.performance_metrics.Test_R2, 4) : '—'}
                hint="Không có đơn vị · mức độ giải thích biến thiên"
                icon={<BarChart3 size={18} />}
              />
            </div>
            <div className="metric-glossary" aria-label="Giải thích metric">
              <p><strong>MAE:</strong> trung bình |Rings thật − Rings dự đoán|, đơn vị vòng; thấp hơn thường tốt hơn.</p>
              <p><strong>RMSE:</strong> căn trung bình bình phương sai số, đơn vị vòng; sai số lớn ảnh hưởng mạnh hơn.</p>
              <p><strong>R²:</strong> hệ số xác định, không có đơn vị; có thể âm khi mô hình kém hơn baseline trung bình trên test.</p>
              <p><strong>Median Baseline:</strong> dùng trung vị Rings của Train làm hằng số dự đoán các mẫu Test.</p>
            </div>
            <p className="chart-caption">
              {modelCard
                ? `Giá trị lấy trực tiếp từ GET /api/model-card; cấu hình runtime K=${modelCard.hyperparameters.n_neighbors}, weights=${modelCard.hyperparameters.weights}.`
                : 'Chưa xác minh được metric và cấu hình runtime vì API đang ngoại tuyến.'}
            </p>
            {modelCard && evaluationReports && !apiMatchesReport ? (
              <div className="notice-box">
                Runtime API hiện không khớp metadata/bảng kết quả đã lưu trong repository. KPI trên
              vẫn là dữ liệu API; các biểu đồ bên dưới là artifact tĩnh, không đồng nhất với
              cấu hình API hiện tại.
              </div>
            ) : null}
          </section>

          <section className="panel chart-panel experiment-card baseline-experiment">
            <div className="section-header">
              <BarChart3 size={18} />
              <h2>Thí nghiệm 1 · Median Baseline và KNN</h2>
            </div>
            {evaluationReports ? (
              <>
                <div className="baseline-chart-scroll">
                  <div className="chart-box baseline-chart">
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={evaluationReports.modelResults} margin={{ top: 12, right: 18, left: 4, bottom: 8 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey="model" tick={{ fontSize: 12 }} />
                        <YAxis label={{ value: 'MAE (Rings)', angle: -90, position: 'insideLeft' }} />
                        <Tooltip formatter={(value) => [`${Number(value).toFixed(4)} Rings`, 'MAE · Test']} />
                        <Bar dataKey="mae" name="MAE · Test" fill="#1f7ae0" radius={[8, 8, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                <div className="baseline-results-scroll">
                  <table className="baseline-results-table">
                    <thead>
                      <tr>
                        <th scope="col">Mô hình</th>
                        <th scope="col">MAE · Test (Rings)</th>
                        <th scope="col">RMSE · Test (Rings)</th>
                        <th scope="col">R² · Test</th>
                      </tr>
                    </thead>
                    <tbody>
                      {evaluationReports.modelResults.map((row) => (
                        <tr key={row.model}>
                          <th scope="row">{row.model}</th>
                          <td>{formatNumber(row.mae, 4)}</td>
                          <td>{formatNumber(row.rmse, 4)}</td>
                          <td>{formatNumber(row.r2, 4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="insight">
                  <strong>Nhận xét kết quả.</strong> Trên test set, các KNN trong artifact có MAE thấp hơn
                  Median Baseline. Bảng phân biệt rõ KNN không scale và StandardScaler; cấu hình API
                  hiện tại chỉ được xác nhận khi API khớp artifact.
                </p>
                <p className="source-note">Nguồn: {evaluationReports.sources.modelResults}. Metric đánh giá trên test set độc lập.</p>
              </>
            ) : <div className="empty-state">{reportsLoading ? 'Đang tải artifact thực nghiệm…' : reportsError}</div>}
          </section>

          <section className="panel chart-panel experiment-card">
            <div className="section-header">
              <Scale size={18} />
              <h2>Thí nghiệm 2 · StandardScaler và No Scaling</h2>
            </div>
            {evaluationReports ? (
              <>
                <div className="chart-box">
                  <ResponsiveContainer width="100%" height={280}>
                    <BarChart data={evaluationReports.cvAblation} margin={{ top: 12, right: 18, left: 4, bottom: 8 }}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} />
                      <XAxis dataKey="preprocessing" />
                      <YAxis label={{ value: 'CV MAE (Rings)', angle: -90, position: 'insideLeft' }} />
                      <Tooltip formatter={(value) => [`${Number(value).toFixed(4)} Rings`, 'CV MAE · Train']} />
                      <Bar dataKey="cvMae" name="CV MAE · Train" fill="#25a9c3" radius={[8, 8, 0, 0]}>
                        <ErrorBar dataKey="cvStd" width={8} stroke="#0b2a42" />
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="ablation-grid">
                  {evaluationReports.cvAblation.map((row) => (
                    <div className="detail-block" key={row.preprocessing}>
                      <span className="detail-label">{row.preprocessing} · CV trên Train</span>
                      <strong>MAE {formatNumber(row.cvMae, 4)} ± {formatNumber(row.cvStd, 4)}</strong>
                      <span className="detail-subtext">Selected K = {row.bestK} · weights = {row.bestWeights}</span>
                    </div>
                  ))}
                </div>
                <p className="insight">
                  <strong>Nhận xét kết quả.</strong> Trong artifact hiện tại, No Scaling có CV MAE thấp
                  hơn StandardScaler; kết quả này không có nghĩa scaling luôn tốt hơn. Đây là kết quả
                  cross-validation trên train, không phải metric test. Model API đang load StandardScaler
                  với cấu hình riêng hiển thị ở KPI phía trên.
                </p>
                <div className="notice-box">
                  Mã pipeline gọi chọn cấu hình với <code>force_scaling=True</code>; vì vậy cấu hình cuối
                  được chọn trong nhánh StandardScaler dù No Scaling có CV MAE thấp hơn ở bảng hiện tại.
                  Artifact không ghi rõ lý do phương pháp cho lựa chọn này; cần nhóm Model xác nhận nếu được hỏi.
                </div>
                <p className="source-note">Nguồn: {evaluationReports.sources.cvAblation}. Sai số hiển thị là CV_STD qua các fold.</p>
              </>
            ) : <div className="empty-state">{reportsLoading ? 'Đang tải artifact thực nghiệm…' : reportsError}</div>}
          </section>

          <section className="panel chart-panel experiment-card">
            <div className="section-header">
              <Activity size={18} />
              <h2>Thí nghiệm 3 · CV MAE theo K</h2>
            </div>
            {evaluationReports ? (
              <>
                <div className="chart-box tall-chart">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={cvChartData} margin={{ top: 14, right: 18, left: 12, bottom: 28 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="k" type="number" domain={['dataMin', 'dataMax']} allowDecimals={false} tickMargin={8} />
                    <YAxis label={{ value: 'CV MAE (Rings)', angle: -90, position: 'insideLeft' }} />
                    <Tooltip labelFormatter={(value) => `K = ${value}`} formatter={(value, name) => [`${Number(value).toFixed(4)} Rings`, name]} />
                    <Line type="monotone" dataKey="scaled_uniform" name="StandardScaler · uniform" stroke={cvLineColors.scaled_uniform} dot={false} connectNulls={false} />
                      <Line type="monotone" dataKey="scaled_distance" name="StandardScaler · distance" stroke={cvLineColors.scaled_distance} dot={false} connectNulls={false} />
                      <Line type="monotone" dataKey="unscaled_uniform" name="No Scaling · uniform" stroke={cvLineColors.unscaled_uniform} dot={false} connectNulls={false} />
                      <Line type="monotone" dataKey="unscaled_distance" name="No Scaling · distance" stroke={cvLineColors.unscaled_distance} dot={false} connectNulls={false} />
                      {selectedCvResult ? (
                        <ReferenceDot
                          x={selectedCvResult.k}
                          y={selectedCvResult.mae}
                          r={7}
                          fill="#061b2b"
                          stroke="#ffffff"
                          strokeWidth={2}
                          label={{ value: `${modelCard ? 'API' : 'Artifact'} K=${selectedCvResult.k}`, position: 'top', fill: '#061b2b', fontSize: 12 }}
                        />
                      ) : null}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div className="cv-axis-caption">Số hàng xóm K</div>
                <div className="cv-legend" aria-label="Chú giải CV MAE">
                  <span><i style={{ backgroundColor: cvLineColors.scaled_uniform }} />StandardScaler · uniform</span>
                  <span><i style={{ backgroundColor: cvLineColors.scaled_distance }} />StandardScaler · distance</span>
                  <span><i style={{ backgroundColor: cvLineColors.unscaled_uniform }} />No Scaling · uniform</span>
                  <span><i style={{ backgroundColor: cvLineColors.unscaled_distance }} />No Scaling · distance</span>
                  {selectedCvResult ? <span><i className="selected-marker" />Cấu hình {modelCard ? 'API runtime' : 'metadata artifact'} được chọn</span> : null}
                </div>
                <p className="insight">
                  <strong>Nhận xét kết quả.</strong> Mỗi đường biểu diễn một tổ hợp preprocessing và
                  weights từ CV thực. Chấm đậm là điểm CV tương ứng K/weights/scaling của API hiện tại
                  khi cấu hình đó có trong CSV; không nội suy hoặc tự điền các điểm thiếu.
                </p>
                <p className="source-note">Nguồn: {evaluationReports.sources.cvResults}. Từng điểm là MAE cross-validation trên train.</p>
              </>
            ) : <div className="empty-state">{reportsLoading ? 'Đang tải artifact thực nghiệm…' : reportsError}</div>}
          </section>

          <section className="panel chart-panel experiment-card">
            <div className="section-header">
              <Target size={18} />
              <h2>Thí nghiệm 4 · Sai số theo nhóm tuổi</h2>
            </div>
            {evaluationReports ? (
              <>
                <div className="chart-box">
                  <ResponsiveContainer width="100%" height={290}>
                    <BarChart data={evaluationReports.ageGroupMae} margin={{ top: 12, right: 18, left: 4, bottom: 8 }}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} />
                      <XAxis dataKey="group" />
                      <YAxis label={{ value: 'MAE (Rings)', angle: -90, position: 'insideLeft' }} />
                      <Tooltip formatter={(value) => [`${Number(value).toFixed(4)} Rings`, 'MAE · Test']} />
                      <Bar dataKey="mae" name="MAE · Test" fill="#1f7ae0" radius={[8, 8, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <p className="chart-caption">
                  {evaluationReports.ageGroupMae.map((row) => `${row.group}: MAE ${formatNumber(row.mae, 4)} Rings`).join(' · ')}
                  {' · '}Số mẫu theo nhóm không có trong artifact.
                </p>
                <p className="insight">
                  <strong>Nhận xét kết quả.</strong> Nhóm được xác định theo Rings thực tế của mẫu trên
                  tập test (Young ≤ 8, Middle 9–12, Old &gt; 12). Đây là phân tích sai số của Regression,
                  không phải mô hình Classification; số lượng mẫu mỗi nhóm không được suy đoán.
                </p>
                <p className="source-note">Nguồn: {evaluationReports.sources.ageGroupMae}; quy tắc nhóm trong src/evaluate.py.</p>
              </>
            ) : <div className="empty-state">{reportsLoading ? 'Đang tải artifact thực nghiệm…' : reportsError}</div>}
          </section>

          <section className="panel chart-panel experiment-card">
            <div className="section-header">
              <Activity size={18} />
              <h2>Residual · Actual Rings và sai số</h2>
            </div>
            {evaluationReports ? (
              <>
                <figure className="residual-figure">
                  <img
                    src="/model-reports/residual_plot.png"
                    alt="Biểu đồ residual trên test set, trục ngang Actual Rings và trục dọc Residual (Actual trừ Predicted), có đường tham chiếu tại 0."
                  />
                  <figcaption>Ảnh residual gốc xuất từ quy trình đánh giá của dự án.</figcaption>
                </figure>
                <p className="chart-caption">
                  Residual = actual − predicted. Đường ngang màu đỏ biểu thị residual = 0; điểm phía
                  trên biểu thị dự đoán thấp hơn thực tế, phía dưới biểu thị dự đoán cao hơn.
                </p>
                <p className="insight">
                  <strong>Nhận xét kết quả.</strong> Phân bố residual quanh đường 0 giúp quan sát xu
                  hướng lệch và sai số lớn theo Rings thực tế. Giao diện dùng ảnh artifact gốc vì repo
                  không cung cấp bảng actual/predicted theo từng mẫu để dựng lại scatter.
                </p>
                <p className="source-note">Nguồn: {evaluationReports.sources.residualPlot}; ảnh được tạo bởi src/evaluate.py trên test set.</p>
              </>
            ) : <div className="empty-state">{reportsLoading ? 'Đang tải artifact thực nghiệm…' : reportsError}</div>}
          </section>

          <section className="panel section-card experiment-card">
            <div className="section-header">
              <Database size={18} />
              <h2>Model Card</h2>
            </div>
            {modelCard ? (
              <>
                <div className="model-details">
                  <div className="detail-block"><span className="detail-label">Algorithm</span><strong>{modelCard.algorithm}</strong></div>
                  <div className="detail-block"><span className="detail-label">Cấu hình runtime</span><strong>K = {String(modelCard.hyperparameters.n_neighbors)} · weights = {String(modelCard.hyperparameters.weights)}</strong></div>
                  <div className="detail-block"><span className="detail-label">Distance metric</span><strong>{String(modelCard.hyperparameters.metric)}</strong></div>
                  <div className="detail-block"><span className="detail-label">Preprocessing</span><strong>{modelCard.pipeline_steps.join(' → ')}</strong></div>
                  <div className="detail-block"><span className="detail-label">Train / test</span><strong>{evaluationReports ? `${evaluationReports.metadata.totalTrainingSamples.toLocaleString('vi-VN')} / ${evaluationReports.metadata.totalTestSamples.toLocaleString('vi-VN')} mẫu` : '—'}</strong></div>
                  <div className="detail-block"><span className="detail-label">CV folds · random_state</span><strong>{String(modelCard.hyperparameters.cv_folds ?? '—')} · {String(modelCard.hyperparameters.random_state ?? '—')}</strong></div>
                  <div className="detail-block"><span className="detail-label">Intended use</span><strong>{modelCard.intended_use}</strong></div>
                  <div className="detail-block"><span className="detail-label">Dataset source</span><strong>{modelCard.dataset_source}</strong></div>
                </div>
                <div className="limitations-list">
                  <h3>Giới hạn</h3>
                  <ul>{safeLimitations.map((limitation) => <li key={limitation}>{limitation}</li>)}</ul>
                </div>
                <div className="technical-glossary">
                  <p><strong>StandardScaler:</strong> chuẩn hóa 7 biến liên tục; <strong>OneHotEncoder:</strong> mã hóa Sex dạng phân loại.</p>
                  <p><strong>weights=distance:</strong> láng giềng gần hơn đóng góp trọng số lớn hơn trong dự đoán; distance metric là Euclidean (Minkowski p=2).</p>
                  <p><strong>5-fold CV:</strong> đánh giá cấu hình bằng các fold tách từ Train; Test được giữ riêng cho đánh giá cuối.</p>
                </div>
              </>
            ) : evaluationReports ? (
              <>
                <div className="notice-box">
                  API ngoại tuyến: dưới đây là metadata của artifact trên đĩa, không thể xác nhận đây là
                  cấu hình mà server đang load.
                </div>
                <div className="model-details">
                  <div className="detail-block"><span className="detail-label">Algorithm</span><strong>{evaluationReports.metadata.pipelineSteps.at(-1)}</strong></div>
                  <div className="detail-block"><span className="detail-label">K · weights · metric</span><strong>{String(evaluationReports.metadata.hyperparameters.n_neighbors)} · {String(evaluationReports.metadata.hyperparameters.weights)} · {String(evaluationReports.metadata.hyperparameters.metric)}</strong></div>
                  <div className="detail-block"><span className="detail-label">Preprocessing</span><strong>{evaluationReports.metadata.pipelineSteps.join(' → ')}</strong></div>
                  <div className="detail-block"><span className="detail-label">Train / test · CV · seed</span><strong>{evaluationReports.metadata.totalTrainingSamples} / {evaluationReports.metadata.totalTestSamples} · {String(evaluationReports.metadata.hyperparameters.cv_folds)} folds · {String(evaluationReports.metadata.hyperparameters.random_state)}</strong></div>
                  <div className="detail-block"><span className="detail-label">Intended use</span><strong>{evaluationReports.metadata.intendedUse}</strong></div>
                  <div className="detail-block"><span className="detail-label">Dataset source</span><strong>{evaluationReports.metadata.datasetSource}</strong></div>
                </div>
                <div className="limitations-list">
                  <h3>Giới hạn</h3>
                  <ul>{safeLimitations.map((limitation) => <li key={limitation}>{limitation}</li>)}</ul>
                </div>
                <div className="technical-glossary">
                  <p><strong>StandardScaler:</strong> chuẩn hóa 7 biến liên tục; <strong>OneHotEncoder:</strong> mã hóa Sex dạng phân loại.</p>
                  <p><strong>weights=distance:</strong> láng giềng gần hơn đóng góp trọng số lớn hơn trong dự đoán; distance metric là Euclidean (Minkowski p=2).</p>
                  <p><strong>5-fold CV:</strong> đánh giá cấu hình bằng các fold tách từ Train; Test được giữ riêng cho đánh giá cuối.</p>
                </div>
              </>
            ) : <div className="empty-state">{reportsLoading ? 'Đang tải metadata…' : reportsError}</div>}
          </section>

          <section className="panel section-card">
            <div className="section-header">
              <Waves size={18} />
              <h2>Phạm vi biến đầu vào</h2>
            </div>

            {featureEntries.length > 0 ? (
              <div className="range-table-wrap">
                <table className="range-table">
                  <thead>
                    <tr>
                      <th>Đặc trưng</th>
                      <th>Ý nghĩa</th>
                      <th>Miền quan sát Train</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>Sex</td>
                      <td>{featureDescriptions.Sex}</td>
                      <td>{featureRanges?.valid_sex_values.join(' · ') ?? 'M · F · I'} · phân loại</td>
                    </tr>
                    {featureEntries.map(([feature, stats]) => (
                      <tr key={feature}>
                        <td>{feature}</td>
                        <td>{featureDescriptions[feature]}</td>
                        <td>{formatNumber(stats.physical_min, 2)}–{formatNumber(stats.physical_max, 2)} {stats.physical_unit}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="empty-state">Đang tải miền giá trị…</div>
            )}
          </section>
        </main>
      ) : null}
    </div>
  )
}

export default App
