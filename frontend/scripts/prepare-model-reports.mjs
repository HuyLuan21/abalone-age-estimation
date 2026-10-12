import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const frontendDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const repositoryDir = path.resolve(frontendDir, '..')
const reportsDir = path.join(repositoryDir, 'reports')
const publicDir = path.join(frontendDir, 'public', 'model-reports')

function parseCsv(text, filename) {
  const lines = text.trim().split(/\r?\n/)
  const parseRow = (line) => {
    const values = []
    let value = ''
    let quoted = false

    for (let index = 0; index < line.length; index += 1) {
      const character = line[index]
      if (character === '"' && line[index + 1] === '"' && quoted) {
        value += '"'
        index += 1
      } else if (character === '"') {
        quoted = !quoted
      } else if (character === ',' && !quoted) {
        values.push(value)
        value = ''
      } else {
        value += character
      }
    }

    values.push(value)
    return values
  }

  const headers = parseRow(lines[0])
  return lines.slice(1).filter(Boolean).map((line) => {
    const values = parseRow(line)
    if (values.length !== headers.length) {
      throw new Error(`${filename}: CSV row has ${values.length} fields; expected ${headers.length}.`)
    }
    return Object.fromEntries(headers.map((header, index) => [header, values[index]]))
  })
}

function number(value, context) {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) {
    throw new Error(`${context}: expected a finite number, received "${value}".`)
  }
  return parsed
}

async function readCsv(filename) {
  return parseCsv(await readFile(path.join(reportsDir, filename), 'utf8'), filename)
}

const [modelRows, ablationRows, cvRows, ageRows, removedRows, metadataText, rawDataText] = await Promise.all([
  readCsv('model_results.csv'),
  readCsv('cv_ablation.csv'),
  readCsv('cv_results.csv'),
  readCsv('age_group_mae.csv'),
  readCsv('removed_anomalies.csv'),
  readFile(path.join(repositoryDir, 'models', 'model_metadata.json'), 'utf8'),
  readFile(path.join(repositoryDir, 'data', 'raw', 'abalone.data'), 'utf8'),
])
const metadata = JSON.parse(metadataText)
const rawSampleCount = rawDataText.trim().split(/\r?\n/).filter(Boolean).length
const removedSampleCount = removedRows.length
const cleanedSampleCount = rawSampleCount - removedSampleCount
const trainingSampleCount = number(metadata.total_training_samples, 'model_metadata.json total_training_samples')
const testSampleCount = number(metadata.total_test_samples, 'model_metadata.json total_test_samples')
const featureStatistics = Object.fromEntries(
  Object.entries(metadata.feature_statistics ?? {}).map(([feature, stats]) => [
    feature,
    {
      description: stats.desc,
      min: number(stats.min, `feature_statistics ${feature} min`) * 200,
      max: number(stats.max, `feature_statistics ${feature} max`) * 200,
      mean: number(stats.mean, `feature_statistics ${feature} mean`) * 200,
      physicalUnit: feature.toLowerCase().includes('weight') ? 'g' : 'mm',
    },
  ]),
)

if (cleanedSampleCount !== trainingSampleCount + testSampleCount) {
  throw new Error(
    `Data-count mismatch: ${rawSampleCount} raw - ${removedSampleCount} removed = ${cleanedSampleCount} cleaned, but metadata reports ${trainingSampleCount} train + ${testSampleCount} test.`,
  )
}

const expectedGroups = ['Young', 'Middle', 'Old']
const ageGroupMae = ageRows.map((row) => ({
  group: row.AgeGroup,
  mae: number(row['0'], `age_group_mae.csv ${row.AgeGroup}`),
}))
if (expectedGroups.some((group) => !ageGroupMae.some((row) => row.group === group))) {
  throw new Error('age_group_mae.csv is missing one or more expected age groups.')
}

const modelResults = modelRows.map((row) => ({
  model: row.Model,
  mae: number(row.MAE, `model_results.csv ${row.Model} MAE`),
  rmse: number(row.RMSE, `model_results.csv ${row.Model} RMSE`),
  r2: number(row.R2, `model_results.csv ${row.Model} R2`),
}))
const cvAblation = ablationRows.map((row) => ({
  preprocessing: row.Preprocessing,
  bestK: number(row.Best_K, `cv_ablation.csv ${row.Preprocessing} K`),
  bestWeights: row.Best_Weights,
  cvMae: number(row.CV_MAE, `cv_ablation.csv ${row.Preprocessing} CV_MAE`),
  cvStd: number(row.CV_STD, `cv_ablation.csv ${row.Preprocessing} CV_STD`),
}))
const cvResults = cvRows.map((row) => ({
  scaling: row.Scaling,
  withScaling: row.with_scaling.toLowerCase() === 'true',
  weights: row.Weights,
  k: number(row.K, 'cv_results.csv K'),
  mae: number(row.MAE, 'cv_results.csv MAE'),
  maeStd: number(row.MAE_STD, 'cv_results.csv MAE_STD'),
}))

if (modelResults.length === 0 || cvAblation.length === 0 || cvResults.length === 0) {
  throw new Error('One or more required model report CSV files contain no result rows.')
}

const payload = {
  sources: {
    modelResults: 'reports/model_results.csv',
    cvAblation: 'reports/cv_ablation.csv',
    cvResults: 'reports/cv_results.csv',
    ageGroupMae: 'reports/age_group_mae.csv',
    metadata: 'models/model_metadata.json',
    residualPlot: 'reports/figures/residual_plot.png',
    removedAnomalies: 'reports/removed_anomalies.csv',
    rawDataset: 'data/raw/abalone.data',
  },
  dataAudit: {
    rawSamples: rawSampleCount,
    removedSamples: removedSampleCount,
    cleanedSamples: cleanedSampleCount,
    trainingSamples: trainingSampleCount,
    testSamples: testSampleCount,
  },
  modelResults,
  cvAblation,
  cvResults,
  ageGroupMae,
  metadata: {
    totalTrainingSamples: trainingSampleCount,
    totalTestSamples: testSampleCount,
    hyperparameters: metadata.hyperparameters,
    pipelineSteps: metadata.pipeline_steps,
    intendedUse: metadata.intended_use,
    limitations: metadata.limitations,
    datasetSource: metadata.dataset_source,
    featureStatistics,
  },
}

await mkdir(publicDir, { recursive: true })
await writeFile(path.join(publicDir, 'evaluation.json'), `${JSON.stringify(payload, null, 2)}\n`, 'utf8')
await copyFile(
  path.join(reportsDir, 'figures', 'residual_plot.png'),
  path.join(publicDir, 'residual_plot.png'),
)
