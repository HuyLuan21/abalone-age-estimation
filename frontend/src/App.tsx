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
  ShieldAlert,
  Sparkles,
  Target,
  Waves,
} from 'lucide-react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
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
  Length: number
  Diameter: number
  Height: number
  Whole_weight: number
  Shucked_weight: number
  Viscera_weight: number
  Shell_weight: number
}

type MetricCardProps = {
  label: string
  value: string
  hint: string
  icon: ReactNode
  tone?: 'primary' | 'success' | 'neutral'
}

const defaultForm: FormState = {
  Sex: 'M',
  Length: 0.455,
  Diameter: 0.365,
  Height: 0.095,
  Whole_weight: 0.514,
  Shucked_weight: 0.2245,
  Viscera_weight: 0.101,
  Shell_weight: 0.15,
}

const API_BASE = '/api'

const samplePresets: Record<'standard' | 'physical' | 'ood', FormState> = {
  standard: { ...defaultForm },
  physical: {
    Sex: 'M',
    Length: 91,
    Diameter: 73,
    Height: 19,
    Whole_weight: 102.8,
    Shucked_weight: 44.9,
    Viscera_weight: 20.2,
    Shell_weight: 30,
  },
  ood: {
    Sex: 'F',
    Length: 0.95,
    Diameter: 0.75,
    Height: 0.35,
    Whole_weight: 3.5,
    Shucked_weight: 1.6,
    Viscera_weight: 0.85,
    Shell_weight: 1.1,
  },
}

const metricColors = ['#1f7ae0', '#25c6d7', '#12a561']

const formatNumber = (value: number | undefined, digits = 3) => {
  if (value === undefined || Number.isNaN(value)) return '—'
  return Number(value).toFixed(digits)
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
  const [form, setForm] = useState<FormState>(defaultForm)
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

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
      } catch (loadError) {
        setError(loadError instanceof Error ? loadError.message : 'Lỗi không xác định khi tải dữ liệu.')
      } finally {
        setLoading(false)
      }
    }

    void loadDashboard()
  }, [])

  const metricsChartData = useMemo(() => {
    if (!modelCard) return []

    return [
      { name: 'MAE', value: modelCard.performance_metrics.Test_MAE ?? 0 },
      { name: 'RMSE', value: modelCard.performance_metrics.Test_RMSE ?? 0 },
      { name: 'R²', value: modelCard.performance_metrics.Test_R2 ?? 0 },
    ]
  }, [modelCard])

  const ageGroupChartData = useMemo(() => {
    if (!modelCard) return []

    return Object.entries(modelCard.age_group_mae).map(([name, value]) => ({
      name,
      value,
    }))
  }, [modelCard])

  const featureEntries = useMemo(
    () => (featureRanges ? Object.entries(featureRanges.numeric_ranges) : []),
    [featureRanges],
  )

  const handleFieldChange = (field: keyof FormState, rawValue: string) => {
    const nextValue = field === 'Sex' ? (rawValue as FormState['Sex']) : Number(rawValue)
    setForm((previous) => ({ ...previous, [field]: nextValue }))
  }

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setSubmitting(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      })

      if (!response.ok) {
        const payload = await response.json().catch(() => null)
        let errorMsg = 'Không thể thực hiện dự đoán với dữ liệu hiện tại.'
        if (typeof payload?.detail === 'string') {
          errorMsg = payload.detail
        } else if (Array.isArray(payload?.detail)) {
          errorMsg = payload.detail
            .map((item: any) => {
              if (typeof item === 'string') return item
              const field = Array.isArray(item?.loc) ? item.loc.filter((l: any) => l !== 'body').join('.') : ''
              const msg = item?.msg?.replace(/^Value error,\s*/i, '') || JSON.stringify(item)
              return field ? `${field}: ${msg}` : msg
            })
            .join('; ')
        } else if (payload?.message) {
          errorMsg = String(payload.message)
        }
        throw new Error(errorMsg)
      }

      const result = (await response.json()) as PredictionResponse
      setPrediction(result)
      setActiveView('predict')
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : 'Lỗi khi thực hiện dự đoán.')
    } finally {
      setSubmitting(false)
    }
  }

  const loadPreset = (key: 'standard' | 'physical' | 'ood') => {
    setForm(samplePresets[key])
    setPrediction(null)
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

        <div className="status-pill">
          <span className="pulse" />
          {loading ? 'Đang đồng bộ' : 'API sẵn sàng'}
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
              value={featureRanges ? 'Abalone' : 'Đang tải'}
              hint="Nguồn: UCI ML Repository"
              icon={<Database size={18} />}
              tone="primary"
            />
            <MetricCard
              label="Mẫu đào tạo"
              value={featureRanges ? featureRanges.total_training_samples.toLocaleString('vi-VN') : '—'}
              hint="80% dữ liệu huấn luyện"
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
              hint="Sai số tuyệt đối trung bình"
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
                <h2>Pipeline</h2>
              </div>
              <ol className="pipeline-list">
                {[
                  'Abalone Dataset',
                  'Data Preparation',
                  'Train/Test Split',
                  'Preprocessing Pipeline',
                  'Cross-validation',
                  'KNN Regression',
                  'Evaluation',
                  'Prediction API',
                ].map((step) => (
                  <li key={step}>{step}</li>
                ))}
              </ol>
            </article>
          </section>

          <section className="panel section-card">
            <div className="section-header">
              <Activity size={18} />
              <h2>8 đặc trưng chính</h2>
            </div>

            <div className="feature-grid">
              {featureEntries.length > 0 ? (
                featureEntries.map(([feature, stats]) => (
                  <div key={feature} className="feature-card">
                    <div className="feature-name">{feature}</div>
                    <div className="feature-meta">{stats.description}</div>
                    <div className="feature-range">
                      {stats.min}–{stats.max} {stats.unit}
                    </div>
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
              {Object.entries(samplePresets).map(([key]) => (
                <button
                  key={key}
                  type="button"
                  className="preset-button"
                  onClick={() => loadPreset(key as 'standard' | 'physical' | 'ood')}
                >
                  {key === 'standard' ? 'Mẫu chuẩn' : key === 'physical' ? 'Mẫu vật lý' : 'Mẫu OOD'}
                </button>
              ))}
            </div>

            <form className="prediction-form" onSubmit={handleSubmit}>
              <div className="form-row">
                <label>
                  <span>Giới tính</span>
                  <select value={form.Sex} onChange={(event) => handleFieldChange('Sex', event.target.value)}>
                    <option value="M">M</option>
                    <option value="F">F</option>
                    <option value="I">I</option>
                  </select>
                </label>
              </div>

              <div className="form-grid">
                {[
                  ['Length', 'Length'],
                  ['Diameter', 'Diameter'],
                  ['Height', 'Height'],
                  ['Whole_weight', 'Whole weight'],
                  ['Shucked_weight', 'Shucked weight'],
                  ['Viscera_weight', 'Viscera weight'],
                  ['Shell_weight', 'Shell weight'],
                ].map(([field, label]) => (
                  <label key={field}>
                    <span>{label}</span>
                    <input
                      type="number"
                      step="0.0001"
                      value={form[field as keyof FormState]}
                      onChange={(event) => handleFieldChange(field as keyof FormState, event.target.value)}
                    />
                  </label>
                ))}
              </div>

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
                  <span className={`badge ${prediction.age_group.toLowerCase()}`}>{prediction.age_group}</span>
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
                  <div className="interval-title">Khoảng tham chiếu láng giềng</div>
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
                </div>

                {prediction.warnings.length > 0 ? (
                  <div className="warning-list">
                    {prediction.warnings.map((warning) => (
                      <div key={warning} className="warning-item">
                        <ShieldAlert size={14} />
                        {warning}
                      </div>
                    ))}
                  </div>
                ) : null}

                <div className="neighbors-wrap">
                  <h3>Láng giềng gần nhất</h3>
                  <div className="neighbor-list">
                    {prediction.nearest_neighbors.map((neighbor, index) => (
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
          <section className="panel chart-panel">
            <div className="section-header">
              <BarChart3 size={18} />
              <h2>Hiệu năng mô hình</h2>
            </div>

            {modelCard ? (
              <div className="chart-box">
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={metricsChartData}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="value" radius={[8, 8, 0, 0]}>
                      {metricsChartData.map((entry, index) => (
                        <Cell key={entry.name} fill={metricColors[index % metricColors.length]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="empty-state">Đang tải thống kê hiệu năng…</div>
            )}
          </section>

          <section className="panel chart-panel">
            <div className="section-header">
              <Activity size={18} />
              <h2>MAE theo nhóm tuổi</h2>
            </div>

            {modelCard ? (
              <div className="chart-box">
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={ageGroupChartData}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="value" radius={[8, 8, 0, 0]} fill="#1d6ad8" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="empty-state">Đang tải dữ liệu nhóm tuổi…</div>
            )}
          </section>

          <section className="panel section-card">
            <div className="section-header">
              <Database size={18} />
              <h2>Thông tin mô hình</h2>
            </div>

            {modelCard ? (
              <div className="model-details">
                <div className="detail-block">
                  <span className="detail-label">Thuật toán</span>
                  <strong>{modelCard.algorithm}</strong>
                </div>
                <div className="detail-block">
                  <span className="detail-label">Nguồn dữ liệu</span>
                  <strong>{modelCard.dataset_source}</strong>
                </div>
                <div className="detail-block">
                  <span className="detail-label">K</span>
                  <strong>{String(modelCard.hyperparameters.n_neighbors)}</strong>
                </div>
                <div className="detail-block">
                  <span className="detail-label">Weights</span>
                  <strong>{String(modelCard.hyperparameters.weights)}</strong>
                </div>
              </div>
            ) : (
              <div className="empty-state">Đang tải metadata mô hình…</div>
            )}
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
                      <th>Biến</th>
                      <th>Min</th>
                      <th>Mean</th>
                      <th>Max</th>
                      <th>Đơn vị</th>
                    </tr>
                  </thead>
                  <tbody>
                    {featureEntries.map(([feature, stats]) => (
                      <tr key={feature}>
                        <td>{feature}</td>
                        <td>{formatNumber(stats.min, 3)}</td>
                        <td>{formatNumber(stats.mean, 3)}</td>
                        <td>{formatNumber(stats.max, 3)}</td>
                        <td>{stats.unit}</td>
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
