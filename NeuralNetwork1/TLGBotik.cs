using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Drawing;
using Telegram.Bot;
using Telegram.Bot.Exceptions;
using Telegram.Bot.Extensions.Polling;
using Telegram.Bot.Types;
using Telegram.Bot.Types.Enums;

namespace NeuralNetwork1
{
    public class TLGBotik
    {
        private TelegramBotClient botik = null;
        private readonly UpdateTLGMessages formUpdater;
        private BaseNetwork network = null;
        private AIMLBotik aimlBot = null;
        private DatasetProcessor datasetProcessor = null;
        private readonly CancellationTokenSource cts = new CancellationTokenSource();

        public TLGBotik(BaseNetwork net, AIMLBotik aiml, DatasetProcessor dataset, UpdateTLGMessages updater)
        {
            formUpdater = updater;
            network = net;
            aimlBot = aiml;
            datasetProcessor = dataset;

            var botKey = System.IO.File.ReadAllText("botkey.txt");
            botik = new TelegramBotClient(botKey);
        }

        public void SetNet(BaseNetwork net)
        {
            network = net;
            formUpdater("Нейросеть обновлена!");
        }

        private async Task HandleUpdateMessageAsync(ITelegramBotClient botClient, Update update, CancellationToken cancellationToken)
        {
            var message = update.Message;
            if (message == null) return;

            var chatId = message.Chat.Id;
            var userName = message.Chat.FirstName ?? "Друг";

            formUpdater($"Получено сообщение от {userName}: {message.Type}");

            // Обработка фотографий
            if (message.Type == MessageType.Photo)
            {
                await ProcessPhotoMessage(message, chatId, userName, cancellationToken);
                return;
            }

            // Обработка текста через AIML
            if (message.Type == MessageType.Text)
            {
                var messageText = message.Text;
                formUpdater($"Текст: {messageText}");

                // Проверяем команду /start
                if (messageText.ToLower().StartsWith("/start"))
                {
                    messageText = "START";
                }

                string response = aimlBot.Talk(chatId, userName, messageText);
                await botik.SendTextMessageAsync(chatId, response, cancellationToken: cancellationToken);
                formUpdater($"Ответ: {response}");
                return;
            }

            // Другие типы сообщений
            if (message.Type == MessageType.Video)
            {
                string response = aimlBot.Talk(chatId, userName, "Видео");
                await botik.SendTextMessageAsync(chatId, response, cancellationToken: cancellationToken);
                return;
            }

            if (message.Type == MessageType.Audio)
            {
                string response = aimlBot.Talk(chatId, userName, "Аудио");
                await botik.SendTextMessageAsync(chatId, response, cancellationToken: cancellationToken);
                return;
            }
        }

        /// <summary>
        /// Обработка фотографии — распознавание буквы Морзе
        /// </summary>
        private async Task ProcessPhotoMessage(Message message, long chatId, string userName, CancellationToken cancellationToken)
        {
            try
            {
                formUpdater("Загрузка изображения...");

                // Получаем файл фото
                var photoId = message.Photo.Last().FileId;
                Telegram.Bot.Types.File fl = await botik.GetFileAsync(photoId, cancellationToken: cancellationToken);

                var imageStream = new MemoryStream();
                await botik.DownloadFileAsync(fl.FilePath, imageStream, cancellationToken: cancellationToken);
                imageStream.Seek(0, SeekOrigin.Begin);

                // Создаём Bitmap
                using (var img = Image.FromStream(imageStream))
                using (var bitmap = new Bitmap(img))
                {
                    // Масштабируем до 200x200
                    using (var resized = new Bitmap(bitmap, new Size(200, 200)))
                    {
                        // Создаём Sample и распознаём
                        Sample sample = datasetProcessor.getSample(resized);
                        LetterType result = network.Predict(sample);

                        // Получаем строковое представление
                        string letterStr = DatasetProcessor.LetterTypeToString(result);
                        string topicStr = DatasetProcessor.LetterTypeToAIMLTopic(result);

                        // Устанавливаем топик в AIML для продолжения разговора
                        aimlBot.SetTopic(chatId, userName, topicStr);

                        // Формируем ответ
                        string response = GetRecognitionResponse(result, letterStr, userName);

                        await botik.SendTextMessageAsync(chatId, response, cancellationToken: cancellationToken);
                        formUpdater($"Распознано: {letterStr}");
                    }
                }
            }
            catch (Exception ex)
            {
                formUpdater($"Ошибка: {ex.Message}");
                await botik.SendTextMessageAsync(chatId,
                    "Извини, не удалось обработать изображение. Попробуй ещё раз!",
                    cancellationToken: cancellationToken);
            }
        }

        /// <summary>
        /// Генерация ответа после распознавания буквы
        /// </summary>
        private string GetRecognitionResponse(LetterType type, string letter, string userName)
        {
            switch (type)
            {
                case LetterType.A:
                    return $"🔍 {userName}, я распознал букву А (·−)!\nЭто точка и тире. Хочешь узнать что-нибудь интересное об этой букве?";
                case LetterType.G:
                    return $"🔍 {userName}, это буква Г (−−·)!\nДва тире и точка. Спроси меня подробнее!";
                case LetterType.E:
                    return $"🔍 {userName}, это буква Е (·)!\nВсего одна точка — самый короткий код! Хочешь узнать почему?";
                case LetterType.Z:
                    return $"🔍 {userName}, я вижу букву З (−−··)!\nДва тире и две точки. Могу рассказать интересный факт!";
                case LetterType.N:
                    return $"🔍 {userName}, это буква Н (−·)!\nТире и точка. Спроси меня о ней!";
                case LetterType.P:
                    return $"🔍 {userName}, распознал букву П (·−−·)!\nСимметричный код! Хочешь узнать больше?";
                case LetterType.T:
                    return $"🔍 {userName}, это буква Т (−)!\nОдно тире — проще некуда! Спроси подробнее!";
                case LetterType.TS:
                    return $"🔍 {userName}, я вижу букву Ц (−·−·)!\nРитмичное чередование. Расскажу интересное!";
                case LetterType.SH:
                    return $"🔍 {userName}, это буква Ш (−−−−)!\nЧетыре тире — самый длинный код! Хочешь факт?";
                case LetterType.SOFT:
                    return $"🔍 {userName}, распознал мягкий знак Ь (−··−)!\nИнтересный симметричный код! Спроси о нём!";
                default:
                    return $"🤔 {userName}, не могу точно определить букву. Попробуй прислать более чёткое изображение!";
            }
        }

        private Task HandleErrorAsync(ITelegramBotClient botClient, Exception exception, CancellationToken cancellationToken)
        {
            var apiRequestException = exception as ApiRequestException;
            if (apiRequestException != null)
            {
                formUpdater($"Telegram API Error: [{apiRequestException.ErrorCode}] {apiRequestException.Message}");
            }
            else
            {
                formUpdater($"Error: {exception.Message}");
            }
            return Task.CompletedTask;
        }

        public bool Act()
        {
            try
            {
                botik.StartReceiving(
                    HandleUpdateMessageAsync,
                    HandleErrorAsync,
                    new ReceiverOptions { AllowedUpdates = new[] { UpdateType.Message } },
                    cancellationToken: cts.Token
                );

                var me = botik.GetMeAsync().Result;
                formUpdater($"Бот запущен: @{me.Username}");
                return true;
            }
            catch (Exception e)
            {
                formUpdater($"Ошибка запуска: {e.Message}");
                return false;
            }
        }

        public void Stop()
        {
            cts.Cancel();
        }
    }
}