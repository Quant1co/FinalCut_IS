using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace NeuralNetwork1
{
    public delegate void UpdateTLGMessages(string msg);

    public partial class Form1 : Form
    {
        /// <summary>
        /// AIML чат-бот
        /// </summary>
        private AIMLBotik aimlBot;

        /// <summary>
        /// Telegram бот
        /// </summary>
        private TLGBotik tlgBot;

        /// <summary>
        /// Обработчик датасета
        /// </summary>
        private DatasetProcessor datasetProcessor;

        /// <summary>
        /// Текущая нейросеть
        /// </summary>
        private BaseNetwork currentNetwork = null;

        /// <summary>
        /// Кэш созданных сетей
        /// </summary>
        private Dictionary<string, BaseNetwork> networksCache = new Dictionary<string, BaseNetwork>();

        private const int SensorCount = 400;

        public Form1()
        {
            InitializeComponent();

            // Инициализация компонентов
            datasetProcessor = new DatasetProcessor();
            aimlBot = new AIMLBotik();

            // Настройка UI
            netTypeBox.SelectedIndex = 0;
            datasetProcessor.LetterCount = (int)classCounter.Value;

            // Создание сети по умолчанию
            CreateNetwork();

            // Инициализация Telegram бота
            tlgBot = new TLGBotik(currentNetwork, aimlBot, datasetProcessor, new UpdateTLGMessages(UpdateTLGInfo));

            pictureBox1.Image = Properties.Resources.Title;
        }

        private int[] GetNetworkStructure()
        {
            return netStructureBox.Text.Split(';').Select(int.Parse).ToArray();
        }

        private void CreateNetwork()
        {
            int[] structure = GetNetworkStructure();

            if (structure.Length < 2 || structure[0] != SensorCount || structure[structure.Length - 1] != datasetProcessor.LetterCount)
            {
                MessageBox.Show($"Структура сети должна начинаться с {SensorCount} и заканчиваться на {datasetProcessor.LetterCount}",
                    "Ошибка", MessageBoxButtons.OK);
                return;
            }

            string netType = (string)netTypeBox.SelectedItem;

            if (netType == "Accord.Net Perseptron")
            {
                currentNetwork = new AccordNet(structure);
            }
            else
            {
                currentNetwork = new StudentNetwork(structure);
            }

            currentNetwork.TrainProgress += UpdateLearningInfo;

            if (tlgBot != null)
            {
                tlgBot.SetNet(currentNetwork);
            }
        }

        public void UpdateLearningInfo(double progress, double error, TimeSpan elapsedTime)
        {
            if (progressBar1.InvokeRequired)
            {
                progressBar1.Invoke(new TrainProgressHandler(UpdateLearningInfo), progress, error, elapsedTime);
                return;
            }

            StatusLabel.Text = "Ошибка: " + error.ToString("F6");
            int prgs = (int)Math.Round(progress * 100);
            prgs = Math.Min(100, Math.Max(0, prgs));
            elapsedTimeLabel.Text = "Время: " + elapsedTime.Duration().ToString(@"hh\:mm\:ss\:ff");
            progressBar1.Value = prgs;
        }

        public void UpdateTLGInfo(string message)
        {
            if (TLGUsersMessages.InvokeRequired)
            {
                TLGUsersMessages.Invoke(new UpdateTLGMessages(UpdateTLGInfo), message);
                return;
            }
            TLGUsersMessages.Text += message + Environment.NewLine;
            TLGUsersMessages.SelectionStart = TLGUsersMessages.Text.Length;
            TLGUsersMessages.ScrollToCaret();
        }

        private void pictureBox1_MouseClick(object sender, MouseEventArgs e)
        {
            if (currentNetwork == null) return;

            var sample = datasetProcessor.getSample();
            currentNetwork.Predict(sample.Item1);

            if (sample.Item1.Correct())
                label1.ForeColor = Color.Green;
            else
                label1.ForeColor = Color.Red;

            label1.Text = "Распознано: " + DatasetProcessor.LetterTypeToString(sample.Item1.recognizedClass);
            pictureBox1.Image = sample.Item2;
            pictureBox1.Invalidate();

            // Обновляем выходы сети
            var output = currentNetwork.GetOutput();
            if (output != null)
            {
                label8.Text = String.Join("\n", output.Select((d, i) =>
                    $"{DatasetProcessor.LetterTypeToString((LetterType)i)}: {d:F3}"));
            }
        }

        private async Task TrainNetworkAsync(int trainingSize, int epochs, double acceptableError, bool parallel)
        {
            label1.Text = "Загрузка датасета...";
            label1.ForeColor = Color.Orange;
            groupBox1.Enabled = false;
            pictureBox1.Enabled = false;
            progressBar1.Value = 0;

            try
            {
                double error = await Task.Run(() =>
                {
                    SamplesSet samples = datasetProcessor.getTrainDataset(trainingSize);
                    return currentNetwork.TrainOnDataSet(samples, epochs, acceptableError, parallel);
                });

                label1.Text = "Обучение завершено! Кликните для теста";
                label1.ForeColor = Color.Green;
                StatusLabel.Text = "Ошибка: " + error.ToString("F6");
                StatusLabel.ForeColor = Color.Green;
            }
            catch (Exception ex)
            {
                label1.Text = $"Ошибка: {ex.Message}";
                label1.ForeColor = Color.Red;
            }
            finally
            {
                groupBox1.Enabled = true;
                pictureBox1.Enabled = true;
            }
        }

        private async Task TestNetworkAsync()
        {
            label1.Text = "Тестирование...";
            label1.ForeColor = Color.Orange;
            groupBox1.Enabled = false;

            try
            {
                int testSize = (int)TrainingSizeCounter.Value;

                double accuracy = await Task.Run(() =>
                {
                    SamplesSet samples = datasetProcessor.getTestDataset(testSize);
                    return samples.TestNeuralNetwork(currentNetwork);
                });

                progressBar1.Value = 100;
                StatusLabel.Text = $"Точность: {accuracy * 100:F2}%";
                StatusLabel.ForeColor = accuracy * 100 >= AccuracyCounter.Value ? Color.Green : Color.Red;
                label1.Text = "Тестирование завершено";
                label1.ForeColor = Color.Green;
            }
            catch (Exception ex)
            {
                label1.Text = $"Ошибка: {ex.Message}";
                label1.ForeColor = Color.Red;
            }
            finally
            {
                groupBox1.Enabled = true;
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            _ = TrainNetworkAsync(
                (int)TrainingSizeCounter.Value,
                (int)EpochesCounter.Value,
                (100 - AccuracyCounter.Value) / 100.0,
                parallelCheckBox.Checked
            );
        }

        private void button2_Click(object sender, EventArgs e)
        {
            _ = TestNetworkAsync();
        }

        private void button3_Click(object sender, EventArgs e)
        {
            CreateNetwork();
        }

        private void classCounter_ValueChanged(object sender, EventArgs e)
        {
            datasetProcessor.LetterCount = (int)classCounter.Value;
            var vals = netStructureBox.Text.Split(';');
            vals[vals.Length - 1] = classCounter.Value.ToString();
            netStructureBox.Text = string.Join(";", vals);
        }

        private void button1_Click_1(object sender, EventArgs e)
        {
            var phrase = AIMLInput.Text;
            if (!string.IsNullOrEmpty(phrase))
            {
                string response = aimlBot.Talk(phrase);
                AIMLOutput.Text += $"Вы: {phrase}{Environment.NewLine}";
                AIMLOutput.Text += $"Бот: {response}{Environment.NewLine}{Environment.NewLine}";
                AIMLInput.Clear();
            }
        }

        private void TLGBotOnButton_Click(object sender, EventArgs e)
        {
            if (tlgBot.Act())
            {
                TLGBotOnButton.Enabled = false;
                TLGBotOnButton.Text = "Бот работает";
            }
        }

        private void netTypeBox_SelectedIndexChanged(object sender, EventArgs e)
        {
            // При смене типа сети пересоздаём
        }

        private void netTrainButton_MouseEnter(object sender, EventArgs e)
        {
            infoStatusLabel.Text = "Обучить нейросеть с указанными параметрами";
        }

        private void testNetButton_MouseEnter(object sender, EventArgs e)
        {
            infoStatusLabel.Text = "Тестировать нейросеть на тестовой выборке";
        }

        private void recreateNetButton_MouseEnter(object sender, EventArgs e)
        {
            infoStatusLabel.Text = "Пересоздать сеть с указанной структурой";
        }
    }
}