using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Linq;

namespace NeuralNetwork1
{
    /// <summary>
    /// Типы букв азбуки Морзе
    /// </summary>
    public enum LetterType : byte { SH = 0, N, G, E, P, T, TS, Z, A, SOFT, Undef };

    public class DatasetProcessor
    {
        public static string LetterTypeToString(LetterType type)
        {
            switch (type)
            {
                case LetterType.SH: return "Ш";
                case LetterType.N: return "Н";
                case LetterType.G: return "Г";
                case LetterType.E: return "Е";
                case LetterType.SOFT: return "Ь";
                case LetterType.Z: return "З";
                case LetterType.T: return "Т";
                case LetterType.TS: return "Ц";
                case LetterType.P: return "П";
                case LetterType.A: return "А";
                case LetterType.Undef: return "Неизвестно";
                default: throw new ArgumentOutOfRangeException(nameof(type), type, null);
            }
        }

        /// <summary>
        /// Преобразует LetterType в строку для установки топика в AIML
        /// </summary>
        public static string LetterTypeToAIMLTopic(LetterType type)
        {
            switch (type)
            {
                case LetterType.SH: return "Ш";
                case LetterType.N: return "Н";
                case LetterType.G: return "Г";
                case LetterType.E: return "Е";
                case LetterType.SOFT: return "Ь";
                case LetterType.Z: return "З";
                case LetterType.T: return "Т";
                case LetterType.TS: return "Ц";
                case LetterType.P: return "П";
                case LetterType.A: return "А";
                default: return "";
            }
        }

        private const int ImageSize = 200;
        private const int SensorCount = ImageSize * 2;
        private const string databaseLocation = "..\\..\\dataset";

        private Random random;
        public int LetterCount { get; set; } = 10;
        private Dictionary<LetterType, List<string>> structure;

        public DatasetProcessor()
        {
            random = new Random();
            structure = new Dictionary<LetterType, List<string>>();

            var letterTypes = new[] {
                LetterType.E, LetterType.G, LetterType.N, LetterType.SOFT,
                LetterType.SH, LetterType.A, LetterType.P, LetterType.T,
                LetterType.Z, LetterType.TS
            };

            foreach (var letterType in letterTypes)
            {
                structure[letterType] = new List<string>();
                var dirPath = Path.Combine(databaseLocation, LetterTypeToString(letterType));
                if (Directory.Exists(dirPath))
                {
                    DirectoryInfo d = new DirectoryInfo(dirPath);
                    structure[letterType].AddRange(d.GetFiles("*.jpeg").Select(f => f.FullName));
                    structure[letterType].AddRange(d.GetFiles("*.jpg").Select(f => f.FullName));
                    structure[letterType].AddRange(d.GetFiles("*.png").Select(f => f.FullName));
                }
            }
        }

        /// <summary>
        /// Извлекает признаки из изображения
        /// </summary>
        public double[] ExtractFeatures(Bitmap bitmap)
        {
            double[] input = new double[SensorCount];

            Bitmap resized = bitmap;
            if (bitmap.Width != ImageSize || bitmap.Height != ImageSize)
            {
                resized = new Bitmap(bitmap, new Size(ImageSize, ImageSize));
            }

            using (FastBitmap.FastBitmap fb = new FastBitmap.FastBitmap(resized))
            {
                for (int x = 0; x < ImageSize; x++)
                {
                    int count = 0;
                    for (int y = 0; y < ImageSize; y++)
                    {
                        var color = fb[x, y];
                        if (color.R < 128 || color.G < 128 || color.B < 128)
                        {
                            count++;
                        }
                    }
                    input[x] = count / (double)ImageSize;
                }

                for (int y = 0; y < ImageSize; y++)
                {
                    int count = 0;
                    for (int x = 0; x < ImageSize; x++)
                    {
                        var color = fb[x, y];
                        if (color.R < 128 || color.G < 128 || color.B < 128)
                        {
                            count++;
                        }
                    }
                    input[ImageSize + y] = count / (double)ImageSize;
                }
            }

            if (resized != bitmap)
            {
                resized.Dispose();
            }

            return input;
        }

        private Bitmap AugmentImage(Bitmap original)
        {
            Bitmap result = new Bitmap(original.Width, original.Height);

            using (Graphics g = Graphics.FromImage(result))
            {
                g.Clear(Color.White);
                g.InterpolationMode = InterpolationMode.HighQualityBilinear;

                int shiftX = random.Next(-10, 11);
                int shiftY = random.Next(-10, 11);
                float scale = 0.9f + (float)random.NextDouble() * 0.2f;

                g.TranslateTransform(original.Width / 2f, original.Height / 2f);
                g.ScaleTransform(scale, scale);
                g.TranslateTransform(-original.Width / 2f + shiftX, -original.Height / 2f + shiftY);

                g.DrawImage(original, 0, 0);
            }

            return result;
        }

        public SamplesSet getTestDataset(int count)
        {
            SamplesSet set = new SamplesSet();
            int samplesPerClass = Math.Max(10, count / LetterCount);

            for (int type = 0; type < LetterCount; type++)
            {
                var letterType = (LetterType)type;
                if (!structure.ContainsKey(letterType) || structure[letterType].Count == 0) continue;

                for (int i = 0; i < samplesPerClass; i++)
                {
                    var samplePath = structure[letterType][random.Next(structure[letterType].Count)];
                    using (var bitmap = new Bitmap(samplePath))
                    {
                        double[] input = ExtractFeatures(bitmap);
                        set.AddSample(new Sample(input, LetterCount, letterType));
                    }
                }
            }
            set.shuffle();
            return set;
        }

        public SamplesSet getTrainDataset(int count)
        {
            SamplesSet set = new SamplesSet();
            int samplesPerClass = count / LetterCount;

            for (int type = 0; type < LetterCount; type++)
            {
                var letterType = (LetterType)type;
                if (!structure.ContainsKey(letterType) || structure[letterType].Count == 0) continue;

                for (int i = 0; i < samplesPerClass; i++)
                {
                    var samplePath = structure[letterType][random.Next(structure[letterType].Count)];
                    using (var original = new Bitmap(samplePath))
                    {
                        Bitmap bitmap = random.NextDouble() > 0.5 ? AugmentImage(original) : original;

                        double[] input = ExtractFeatures(bitmap);
                        set.AddSample(new Sample(input, LetterCount, letterType));

                        if (bitmap != original)
                        {
                            bitmap.Dispose();
                        }
                    }
                }
            }
            set.shuffle();
            return set;
        }

        public Tuple<Sample, Bitmap> getSample()
        {
            var type = (LetterType)random.Next(LetterCount);
            if (!structure.ContainsKey(type) || structure[type].Count == 0)
            {
                foreach (var kvp in structure)
                {
                    if (kvp.Value.Count > 0)
                    {
                        type = kvp.Key;
                        break;
                    }
                }
            }

            var samplePath = structure[type][random.Next(structure[type].Count)];
            var bitmap = new Bitmap(samplePath);
            double[] input = ExtractFeatures(bitmap);
            return Tuple.Create(new Sample(input, LetterCount, type), bitmap);
        }

        public Sample getSample(Bitmap bitmap)
        {
            double[] input = ExtractFeatures(bitmap);
            return new Sample(input, LetterCount);
        }
    }
}