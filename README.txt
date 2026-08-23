HOW TO RUN THIS APP
====================

1. Install Python 3.9+ if you don't already have it.

2. Open a terminal in the folder with these files and install the
   required packages:

       pip install -r requirements.txt

3. Start the app:

       streamlit run app.py

4. Your web browser will open automatically to something like
   http://localhost:8501 — that's the app.

5. To use it: type a stock symbol (like AAPL), pick a time range,
   and the charts update automatically. Click the button in the top
   right to switch between English and Mandarin.

TO SHARE WITH YOUR DAD
========================
The easiest way to give him a permanent link (so he doesn't need to
install anything) is Streamlit Community Cloud (free):
  1. Put app.py and requirements.txt in a GitHub repository.
  2. Go to share.streamlit.io, sign in, and point it at your repo.
  3. It gives you a public URL he can bookmark.
