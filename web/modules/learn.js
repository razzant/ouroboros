/** Post-setup field guide. All actions prepare a draft or open an existing page; none sends. */
import { renderPageHeader } from './page_header.js';
import { PAGE_ICONS } from './page_icons.js';

const starters = [
    { title: 'Познакомиться', text: 'Меня зовут <имя>. Я занимаюсь <дело>. Для меня важно <что именно>. Запомни это с моей поправкой, если я что-то изменю.' },
    { title: 'Разобрать файл', text: 'Прочитай приложенный файл целиком и выпиши <что искать>. Отдели факты от предположений и назови, что не удалось проверить.' },
    { title: 'Поставить напоминание', text: 'Напомни мне <дата и время> о <дело>. Скажи, удалось ли сохранить напоминание и что случится, если приложение не работает.' },
    { title: 'Изменить себя аккуратно', text: 'Хочу изменить в Ouroboros <что именно>. Сначала проверь текущий код и границы, предложи план и тесты. Сохрани изменения в отдельной ветке и не публикуй без моего решения.' },
    { title: 'Подготовить issue', text: 'Помоги составить issue для <репозиторий>: воспроизведение <шаги>, ожидаемое и фактическое поведение <разница>. Проверь, не существует ли уже такого issue; сначала покажи черновик.' },
    { title: 'Подготовить PR', text: 'Помоги подготовить PR для <репозиторий> из отдельной чистой копии. Назови базовый SHA, затронутые контракты, проверки и известные ограничения. Перед публикацией покажи финальный diff и результаты независимого ревью.' },
];

const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));

export function appendToDraft(existing, suggestion) {
    const old = String(existing || '');
    return old ? `${old}\n\n${suggestion}` : suggestion;
}

export function initLearn({ showPage, openSettingsTab, openDashboardTab }) {
    const page = document.createElement('section');
    page.id = 'page-learn';
    page.className = 'page learn-page';
    page.innerHTML = `${renderPageHeader({ title: 'Знакомство', icon: PAGE_ICONS.learn })}
        <div class="learn-content">
            <header class="learn-hero">
                <div class="learn-hero-copy">
                <p class="learn-eyebrow">Начни с разговора</p>
                <h2>Знакомься: <span>Ouroboros</span></h2>
                <p>Я работаю с задачами, файлами и инструментами, помню разговоры и могу возвращаться к работе между ними. Но мои возможности зависят от выбранной модели, доступа, бюджета и того, запущено ли приложение. Здесь — карта первых шагов, а не обещание, что всё всегда сработает.</p>
                <div class="learn-hero-actions"><button type="button" data-open="chat" class="btn btn-primary">Открыть чат</button><button type="button" data-open="settings" class="btn btn-secondary">Проверить настройки</button></div>
                </div>
                <div class="learn-orbit" aria-hidden="true"><span class="learn-orbit-core">∞</span><span class="learn-orbit-word learn-orbit-word-top">разговор</span><span class="learn-orbit-word learn-orbit-word-right">действие</span><span class="learn-orbit-word learn-orbit-word-bottom">проверка</span><span class="learn-orbit-word learn-orbit-word-left">память</span></div>
            </header>
            <div class="learn-contents" role="group" aria-label="Разделы знакомства">
                <a href="#learn-start">Начать</a><a href="#learn-work">Что поручить</a><a href="#learn-money">Экономно</a><a href="#learn-change">Изменения</a><a href="#learn-contribute">Issue и PR</a><a href="#learn-limits">Границы</a>
            </div>
            <section id="learn-start" class="learn-section"><h3>Три шага для начала</h3>
                <ol class="learn-steps"><li><strong>Подключи модель.</strong> При первом запуске мастер поможет выбрать аккаунт, модель, ревью и бюджет. Если настройка уже завершена, проверь её в Settings → Accounts и Models.</li>
                <li><strong>Дай конкретную задачу.</strong> Скажи, зачем она нужна, где исходные файлы и как проверить результат. Ниже есть черновики — их отправляешь только ты.</li>
                <li><strong>Проверь результат.</strong> Открой карточку задачи и файлы. Статус, ответ, тесты, ревью и доставленный результат — разные факты.</li></ol></section>
            <section id="learn-work" class="learn-section"><h3>Что поручить</h3><p>Заполни рамку своими словами. Кнопка добавит её к черновику Main Chat, но ничего не отправит и не сотрёт уже набранный текст.</p>
                <div class="learn-templates">${starters.map((item, index) => `<article class="learn-template"><h4>${escapeHtml(item.title)}</h4><label for="learn-template-${index}">Текст задачи</label><textarea id="learn-template-${index}" rows="4">${escapeHtml(item.text)}</textarea><button type="button" data-template="${index}" class="btn btn-secondary">Добавить к черновику →</button></article>`).join('')}</div>
                <p class="learn-feedback" role="status" aria-live="polite"></p></section>
            <section id="learn-money" class="learn-section"><h3>Экономно — это управлять маршрутом</h3><div class="learn-grid">
                <article><h4>Начни с доступного</h4><p>Подписка расходует квоту; API-ключ может тарифицироваться по токенам. Пустая цена в интерфейсе не означает бесплатный вызов. В Accounts проверь подключение, в Models — назначение Main и Light.</p></article>
                <article><h4>Ограничи риск</h4><p>В Settings → Behavior проверь контекст и фоновые задачи, в Dashboard → Costs — учёт расходов. Nano уменьшает рабочее окно, но не отменяет ревью. Предельный бюджет и видимые оценки не гарантируют точный счёт внешнего провайдера.</p></article>
                <article><h4>Делай проверку по масштабу</h4><p>Для простого вопроса не запускай большой рой. Для изменения кода оговори границы и критерий успеха заранее. Не путай несколько успешно прошедших тестов с полной проверкой.</p></article></div>
                <button type="button" class="btn btn-secondary" data-open="costs">Открыть расходы →</button></section>
            <section id="learn-change" class="learn-section"><h3>Менять себя, сохраняя путь назад</h3>
                <p>Я могу читать и менять собственный код, но хорошее поручение называет цель, затронутый контракт, тесты и границу публикации. Работай в отдельной чистой копии или ветке; сравни diff с актуальной базой, проверь связанные документы и вызовы, затем попроси независимую проверку окончательных байтов. Обычный коммит в рабочей ветке Ouroboros — релиз с версией и ревью; внешние PR сохраняют версию до интеграции. Аварийные снимки и механический откат — отдельные исключения.</p>
                <p>Ревью не равно PASS, если рецензент ещё не ответил или его маршрут недоступен. Исправил diff — проверь изменённые байты снова. Слияние и установка — отдельные действия; открытый PR не означает, что изменение уже работает у тебя.</p></section>
            <section id="learn-contribute" class="learn-section"><h3>Как оформить issue или PR</h3><div class="learn-grid">
                <article><h4>Issue: покажи наблюдаемое</h4><p>Версия, система, шаги воспроизведения, ожидаемое и фактическое поведение, безопасный фрагмент лога. Сначала проверь существующие issue. Убери токены, личные данные и пути, которые не хочешь публиковать.</p></article>
                <article><h4>PR: держи границу узкой</h4><p>Одна цель и актуальная upstream-база. Объясни, почему изменение нужно, какие контракты меняет, как проверено и что осталось непроверенным. Попроси меня подготовить текст и проверить GitHub-цель; отправка требует твоего явного решения.</p></article></div>
                <p>GitHub-инструменты зависят от подключённого аккаунта и прав; отказ или отсутствующий инструмент — не опубликованный результат.</p></section>
            <section id="learn-limits" class="learn-section"><h3>Честно о границах</h3><div class="learn-grid">
                <article><h4>Память — не гарантия точности</h4><p>История и заметки сохраняются, но свёртка и поиск могут ошибиться или пропустить контекст. Попроси показать источник и поправь меня, если запись неверна.</p></article>
                <article><h4>Фон требует работающего процесса</h4><p>Пробуждения зависят от настроек, бюджета и запущенного Ouroboros. Закрытие приложения не обещает продолжения и уведомления вне отдельного транспорта.</p></article>
                <article><h4>Инструменты имеют границы</h4><p>Модель не получает доступ ко всем папкам и сервисам по одному обещанию. Проверяй, что разрешено, а результат внешнего действия — по квитанции, а не по моему намерению.</p></article>
                <article><h4>Изображения и экраны</h4><p>Текстовая модель может получить описание вместо пикселей. Для визуальной проверки нужен доступный зрячий маршрут и просмотр реального результата; скриншот сам по себе не проверка.</p></article></div></section>
        </div>`;
    document.getElementById('content').appendChild(page);
    page.addEventListener('click', async (event) => {
        const button = event.target.closest('button');
        if (!button || !page.contains(button)) return;
        if (button.dataset.template !== undefined) {
            const text = page.querySelector(`#learn-template-${button.dataset.template}`)?.value.trim();
            const input = document.querySelector('#page-chat #chat-input');
            const feedback = page.querySelector('.learn-feedback');
            if (!text || !input) { feedback.textContent = 'Не удалось подготовить черновик. Открой Main Chat и попробуй снова.'; return; }
            if (!await showPage('chat')) { feedback.textContent = 'Переход отменён: незавершённые изменения на текущей странице сохранены.'; return; }
            input.value = appendToDraft(input.value, text);
            input.dispatchEvent(new Event('input', { bubbles: true }));
            feedback.textContent = 'Добавлено к черновику Main Chat. Отправка — только после твоего нажатия Send.';
            input.focus();
            return;
        }
        if (button.dataset.open === 'settings') void openSettingsTab('providers');
        else if (button.dataset.open === 'costs') void openDashboardTab('costs');
        else if (button.dataset.open === 'chat') void showPage('chat');
    });
    // In-page links must not mutate the application's one-shot #page route.
    page.querySelectorAll('.learn-contents a').forEach((link) => link.addEventListener('click', (event) => {
        event.preventDefault();
        page.querySelector(link.getAttribute('href'))?.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth' });
    }));
}
