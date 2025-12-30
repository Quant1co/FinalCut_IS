using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace NeuralNetwork1
{
    class Neuron
    {
        public static Func<double, double> activationFunction;
        public static Func<double, double> activationFunctionDerivative;

        public double Output;
        public int layer;
        public double error;

        // Веса связей от предыдущего слоя (0 элемент - bias)
        public double[] weightsToPrevLayer;
        // Предыдущие изменения весов для momentum
        public double[] prevDeltaWeights;

        public void SetInput(double input)
        {
            if (layer == 0)
            {
                Output = input;
                return;
            }
            Output = activationFunction(input);
        }

        public Neuron(int layer, int prevLayerCapacity, Random random)
        {
            this.layer = layer;
            this.error = 0;

            if (layer == -1)
            {
                Output = 1; // Bias
            }

            if (layer < 1)
            {
                weightsToPrevLayer = null;
                prevDeltaWeights = null;
            }
            else
            {
                int weightsCount = prevLayerCapacity + 1;
                weightsToPrevLayer = new double[weightsCount];
                prevDeltaWeights = new double[weightsCount];

                // Xavier инициализация для лучшей сходимости
                double stddev = Math.Sqrt(2.0 / (prevLayerCapacity + 1));
                for (int i = 0; i < weightsCount; i++)
                {
                    // Box-Muller для нормального распределения
                    double u1 = 1.0 - random.NextDouble();
                    double u2 = 1.0 - random.NextDouble();
                    double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                    weightsToPrevLayer[i] = randStdNormal * stddev;
                    prevDeltaWeights[i] = 0;
                }
            }
        }
    }

    public class StudentNetwork : BaseNetwork
    {
        // Гиперпараметры
        private double learningRate = 0.1;
        private double momentum = 0.9;

        // Флаг параллельного расчёта
        private bool useParallel = false;

        private Neuron biasNeuron;
        private List<Neuron[]> layers;

        public StudentNetwork(int[] structure)
        {
            if (structure.Length < 3)
            {
                throw new ArgumentException("Сеть должна иметь минимум 3 слоя");
            }

            // Сигмоида и её производная
            Neuron.activationFunction = x => 1.0 / (1.0 + Math.Exp(-x));
            Neuron.activationFunctionDerivative = y => y * (1.0 - y); // y = sigmoid(x)

            Random random = new Random(42); // Фиксированный seed для воспроизводимости

            biasNeuron = new Neuron(-1, -1, random);
            layers = new List<Neuron[]>();

            for (int layer = 0; layer < structure.Length; layer++)
            {
                layers.Add(new Neuron[structure[layer]]);
                int prevLayerSize = layer == 0 ? -1 : structure[layer - 1];

                for (int i = 0; i < structure[layer]; i++)
                {
                    layers[layer][i] = new Neuron(layer, prevLayerSize, random);
                }
            }
        }

        private void ForwardPropagation(double[] input)
        {
            // Входной слой
            for (int i = 0; i < layers[0].Length; i++)
            {
                layers[0][i].SetInput(input[i]);
            }

            // Скрытые и выходной слои
            for (int layer = 1; layer < layers.Count; layer++)
            {
                var currentLayer = layers[layer];
                var prevLayer = layers[layer - 1];

                if (useParallel)
                {
                    // Параллельный расчёт по нейронам текущего слоя
                    Parallel.For(0, currentLayer.Length, n =>
                    {
                        var neuron = currentLayer[n];
                        double sum = neuron.weightsToPrevLayer[0] * biasNeuron.Output; // Bias

                        for (int p = 0; p < prevLayer.Length; p++)
                        {
                            sum += neuron.weightsToPrevLayer[p + 1] * prevLayer[p].Output;
                        }

                        neuron.SetInput(sum);
                    });
                }
                else
                {
                    // Последовательный расчёт
                    for (int n = 0; n < currentLayer.Length; n++)
                    {
                        var neuron = currentLayer[n];
                        double sum = neuron.weightsToPrevLayer[0] * biasNeuron.Output; // Bias

                        for (int p = 0; p < prevLayer.Length; p++)
                        {
                            sum += neuron.weightsToPrevLayer[p + 1] * prevLayer[p].Output;
                        }

                        neuron.SetInput(sum);
                    }
                }
            }
        }

        private void BackwardPropagation(double[] target)
        {
            // Ошибка выходного слоя
            var outputLayer = layers[layers.Count - 1];

            if (useParallel)
            {
                Parallel.For(0, outputLayer.Length, i =>
                {
                    double output = outputLayer[i].Output;
                    outputLayer[i].error = (target[i] - output) * Neuron.activationFunctionDerivative(output);
                });
            }
            else
            {
                for (int i = 0; i < outputLayer.Length; i++)
                {
                    double output = outputLayer[i].Output;
                    outputLayer[i].error = (target[i] - output) * Neuron.activationFunctionDerivative(output);
                }
            }

            // Обратное распространение ошибки
            for (int layer = layers.Count - 2; layer >= 1; layer--)
            {
                var currentLayer = layers[layer];
                var nextLayer = layers[layer + 1];

                if (useParallel)
                {
                    Parallel.For(0, currentLayer.Length, n =>
                    {
                        double errorSum = 0;
                        for (int k = 0; k < nextLayer.Length; k++)
                        {
                            errorSum += nextLayer[k].error * nextLayer[k].weightsToPrevLayer[n + 1];
                        }
                        currentLayer[n].error = errorSum * Neuron.activationFunctionDerivative(currentLayer[n].Output);
                    });
                }
                else
                {
                    for (int n = 0; n < currentLayer.Length; n++)
                    {
                        double errorSum = 0;
                        for (int k = 0; k < nextLayer.Length; k++)
                        {
                            errorSum += nextLayer[k].error * nextLayer[k].weightsToPrevLayer[n + 1];
                        }
                        currentLayer[n].error = errorSum * Neuron.activationFunctionDerivative(currentLayer[n].Output);
                    }
                }
            }

            // Обновление весов с momentum
            for (int layer = 1; layer < layers.Count; layer++)
            {
                var currentLayer = layers[layer];
                var prevLayer = layers[layer - 1];

                if (useParallel)
                {
                    Parallel.For(0, currentLayer.Length, n =>
                    {
                        var neuron = currentLayer[n];

                        // Bias
                        double delta = learningRate * neuron.error * biasNeuron.Output + momentum * neuron.prevDeltaWeights[0];
                        neuron.weightsToPrevLayer[0] += delta;
                        neuron.prevDeltaWeights[0] = delta;

                        // Остальные веса
                        for (int p = 0; p < prevLayer.Length; p++)
                        {
                            delta = learningRate * neuron.error * prevLayer[p].Output + momentum * neuron.prevDeltaWeights[p + 1];
                            neuron.weightsToPrevLayer[p + 1] += delta;
                            neuron.prevDeltaWeights[p + 1] = delta;
                        }
                    });
                }
                else
                {
                    for (int n = 0; n < currentLayer.Length; n++)
                    {
                        var neuron = currentLayer[n];

                        // Bias
                        double delta = learningRate * neuron.error * biasNeuron.Output + momentum * neuron.prevDeltaWeights[0];
                        neuron.weightsToPrevLayer[0] += delta;
                        neuron.prevDeltaWeights[0] = delta;

                        // Остальные веса
                        for (int p = 0; p < prevLayer.Length; p++)
                        {
                            delta = learningRate * neuron.error * prevLayer[p].Output + momentum * neuron.prevDeltaWeights[p + 1];
                            neuron.weightsToPrevLayer[p + 1] += delta;
                            neuron.prevDeltaWeights[p + 1] = delta;
                        }
                    }
                }
            }
        }

        private double CalculateLoss(double[] target)
        {
            double loss = 0;
            var outputLayer = layers[layers.Count - 1];
            for (int i = 0; i < outputLayer.Length; i++)
            {
                double diff = target[i] - outputLayer[i].Output;
                loss += diff * diff;
            }
            return loss * 0.5;
        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            useParallel = parallel;
            int iterations = 0;
            int maxIterations = 100;

            while (iterations < maxIterations)
            {
                iterations++;
                ForwardPropagation(sample.input);

                if (CalculateLoss(sample.outputVector) <= acceptableError)
                    break;

                BackwardPropagation(sample.outputVector);
            }

            return iterations;
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            useParallel = parallel;
            var start = DateTime.Now;
            double totalError = 0;
            int samplesProcessed = 0;
            int totalSamples = epochsCount * samplesSet.Count;

            for (int epoch = 0; epoch < epochsCount; epoch++)
            {
                double epochError = 0;

                // Перемешиваем данные каждую эпоху
                samplesSet.shuffle();

                for (int i = 0; i < samplesSet.Count; i++)
                {
                    var sample = samplesSet[i];

                    ForwardPropagation(sample.input);
                    epochError += CalculateLoss(sample.outputVector);
                    BackwardPropagation(sample.outputVector);

                    samplesProcessed++;

                    // Обновляем прогресс каждые 50 сэмплов
                    if (i % 50 == 0)
                    {
                        OnTrainProgress(
                            (double)samplesProcessed / totalSamples,
                            epochError / (i + 1),
                            DateTime.Now - start
                        );
                    }
                }

                double avgEpochError = epochError / samplesSet.Count;
                totalError = avgEpochError;

                // Ранняя остановка
                if (avgEpochError <= acceptableError)
                {
                    OnTrainProgress(1.0, avgEpochError, DateTime.Now - start);
                    return avgEpochError;
                }
            }

            OnTrainProgress(1.0, totalError, DateTime.Now - start);
            return totalError;
        }

        protected override double[] Compute(double[] input)
        {
            ForwardPropagation(input);
            return layers[layers.Count - 1].Select(n => n.Output).ToArray();
        }
    }
}