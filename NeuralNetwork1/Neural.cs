using System;
using System.Collections.Generic;
using System.Collections;
using Accord.Math;

namespace NeuralNetwork1
{
    // LetterType определён в DatasetProcessor.cs, здесь не дублируем!

    /// <summary>
    /// Класс для хранения образа
    /// </summary>
    public class Sample
    {
        public double[] input = null;
        public double[] error = null;
        public LetterType actualClass;
        public LetterType recognizedClass;
        public double[] outputVector;

        public Sample(double[] inputValues, int classesCount, LetterType sampleClass = LetterType.Undef)
        {
            input = (double[])inputValues.Clone();
            Output = new double[classesCount];
            if (sampleClass != LetterType.Undef) Output[(int)sampleClass] = 1;

            recognizedClass = LetterType.Undef;
            actualClass = sampleClass;

            outputVector = new double[classesCount];
            for (int i = 0; i < outputVector.Length; i++)
            {
                outputVector[i] = i == (int)actualClass ? 1 : 0;
            }
        }

        public double[] Output { get; private set; }

        public LetterType ProcessPrediction(double[] neuralOutput)
        {
            Output = neuralOutput;
            if (error == null)
                error = new double[Output.Length];

            recognizedClass = 0;
            for (int i = 0; i < Output.Length; ++i)
            {
                error[i] = (Output[i] - (i == (int)actualClass ? 1 : 0));
                if (Output[i] > Output[(int)recognizedClass]) recognizedClass = (LetterType)i;
            }

            return recognizedClass;
        }

        public double EstimatedError()
        {
            double Result = 0;
            for (int i = 0; i < Output.Length; ++i)
                Result += Math.Pow(error[i], 2);
            return Result;
        }

        public bool Correct()
        {
            return actualClass == recognizedClass;
        }

        public override string ToString()
        {
            return $"Actual: {actualClass}, Recognized: {recognizedClass}";
        }
    }

    /// <summary>
    /// Выборка образов
    /// </summary>
    public class SamplesSet : IEnumerable
    {
        public List<Sample> samples = new List<Sample>();

        public void AddSample(Sample image)
        {
            samples.Add(image);
        }

        public int Count => samples.Count;

        public IEnumerator GetEnumerator()
        {
            return samples.GetEnumerator();
        }

        public Sample this[int i]
        {
            get => samples[i];
            set => samples[i] = value;
        }

        public void shuffle()
        {
            samples.Shuffle();
        }

        public double TestNeuralNetwork(BaseNetwork network)
        {
            double correct = 0;
            double wrong = 0;
            foreach (var sample in samples)
            {
                if (sample.actualClass == network.Predict(sample)) ++correct;
                else ++wrong;
            }
            return correct / (correct + wrong);
        }
    }
}