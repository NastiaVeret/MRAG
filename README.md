# MRAG
![image](https://github.com/user-attachments/assets/cd1d9c74-8094-4aa1-9876-97dc6ef840ef)

# Архітектура
1. Першим кроком було стягування новин зі сайту DeepLearning.AI. Я стягнула інформацію по Issues від https://www.deeplearning.ai/the-batch/issue-270 до https://www.deeplearning.ai/the-batch/issue-299. Щоб це зробити я використовувала Web Scraping по html тегах, щоб стягнути всі новини після заголовка News, а саме фотографії, контекст і назву новини.
2. Далі наступним кроком я розбивала на chunks. Точніше я це зробила з 2 спроби. Моя перша спроба була без розбиття на chunks і під час тестування я виявила, що точність була гіршою. Наприклад, я вводила якесь питання, відповідь на яке точно є в новинах, і мені видавало відповідь, що немає відповідного контексту. Тоді я спробувала розбити на chunks за допомогою RecursiveCharacterTextSplitter і точність покращилася. Думаю це завдяки тому, що деякі моделі приймають на вхід обмежену довжину токенів, і можуть обрізати текст і інший не обробляти. А розбиття на менші chunks гарантує, що інформація ніде не пропаде
3. Дальше я перетворила в ембедінги текст за допомогою трансформерів, а саме all-MiniLM-L6-v2. Я її обрала тому, що вона мала і швидко працює навіть на CPU. Для ембедінгу зображень я використовувала CLIP. Про цю модель розповідалося в одному з курсів від DeepLearning.AI, що вона гарно підходить для зображень і для мультимодального пошуку.
4. Далі для того, щоб знаходити схожі елементи серед документів, я використовувала FAISS index. Перший раз я познайомилася з ним під час виконання одної лабораторної і помітила, що він ефективно шукає схожі вектори для запиту.
5. В якості моделі я обрала модель Gemini gemini-1.5-flash, бо вона має високу швидкість, але може мати проблеми з точністю.
6. Під час проходження курсу від DeepLearning.AI, я власне там і дізналася про те, як саме потрібно оцінювати RAG системи - за допомогою Context relevance, groundedness and answer relevance. Власне під час оцінювання якості надання відповіді і контексту, я орієнтувалася на ці 3 метрики.

# Setup
1. Спочатку необхідно склонувати репозиторій
2. В ньому буде jupyter notebook, де можна проглянути усі кроки детально
3. Потрібно в консолі виконати команду streamlit run retrieve_info.py 
