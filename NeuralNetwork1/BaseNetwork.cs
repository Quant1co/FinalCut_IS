using System;

namespace NeuralNetwork1
{
    public delegate void TrainProgressHandler(double progress, double error, TimeSpan time);

    /// <summary>
    /// Базовый класс для реализации нейросетей
    /// </summary>
    public abstract class BaseNetwork
    {
        // Событие обновления прогресса обучения
        public event TrainProgressHandler TrainProgress;

        /// <summary>
        /// Обучение сети одному образу
        /// </summary>
        public abstract int Train(Sample sample, double acceptableError, bool parallel);

        /// <summary>
        /// Обучение сети на датасете
        /// </summary>
        public abstract double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel);

        /// <summary>
        /// Подсчёт результата работы сети
        /// </summary>
        protected abstract double[] Compute(double[] input);

        /// <summary>
        /// Распознавание образа
        /// </summary>
        public LetterType Predict(Sample sample)
        {
            return sample.ProcessPrediction(Compute(sample.input));
        }

        /// <summary>
        /// Получить выходной вектор последнего вычисления
        /// </summary>
        public double[] GetOutput()
        {
            return lastOutput;
        }

        protected double[] lastOutput;

        /// <summary>
        /// Оповещение подписчиков о прогрессе
        /// </summary>
        protected virtual void OnTrainProgress(double progress, double error, TimeSpan time)
        {
            TrainProgress?.Invoke(progress, error, time);
        }
    }
}