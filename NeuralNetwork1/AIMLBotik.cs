using System;
using System.Collections.Generic;
using AIMLbot;

namespace NeuralNetwork1
{
    /// <summary>
    /// AIML чат-бот с поддержкой нескольких пользователей и установки топика
    /// </summary>
    public class AIMLBotik
    {
        private readonly Bot bot;
        // Явно указываем AIMLbot.User чтобы избежать конфликта с Telegram.Bot.Types.User
        private readonly Dictionary<long, AIMLbot.User> users = new Dictionary<long, AIMLbot.User>();

        public AIMLBotik()
        {
            bot = new Bot();
            bot.loadSettings();
            bot.isAcceptingUserInput = false;
            bot.loadAIMLFromFiles();
            bot.isAcceptingUserInput = true;
        }

        /// <summary>
        /// Получить или создать пользователя
        /// </summary>
        private AIMLbot.User GetOrCreateUser(long userId, string userName)
        {
            if (!users.ContainsKey(userId))
            {
                var user = new AIMLbot.User(userId.ToString(), bot);
                users.Add(userId, user);

                // Автоматически представляемся при первом контакте
                if (!string.IsNullOrEmpty(userName))
                {
                    Request r = new Request($"Меня зовут {userName}", user, bot);
                    bot.Chat(r);
                }
            }
            return users[userId];
        }

        /// <summary>
        /// Обработка текстового сообщения
        /// </summary>
        public string Talk(long userId, string userName, string phrase)
        {
            var user = GetOrCreateUser(userId, userName);
            Request request = new Request(phrase, user, bot);
            Result result = bot.Chat(request);
            return result.Output;
        }

        /// <summary>
        /// Установка топика для пользователя (после распознавания буквы)
        /// </summary>
        public void SetTopic(long userId, string userName, string topic)
        {
            var user = GetOrCreateUser(userId, userName);
            user.Predicates.updateSetting("topic", topic);
        }

        /// <summary>
        /// Локальный чат (для тестирования на форме)
        /// </summary>
        public string Talk(string phrase)
        {
            return Talk(0, "LocalUser", phrase);
        }

        /// <summary>
        /// Установка топика локально
        /// </summary>
        public void SetTopic(string topic)
        {
            SetTopic(0, "LocalUser", topic);
        }
    }
}