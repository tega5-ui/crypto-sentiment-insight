import requests

# خريطة تحويل الرموز إلى CoinGecko IDs
symbol_map = {
    "BTC-USD": "bitcoin",
    "ETH-USD": "ethereum",
    "BNB-USD": "binancecoin",
    "ADA-USD": "cardano",
    "SOL-USD": "solana",
    "XRP-USD": "ripple",
    "DOGE-USD": "dogecoin",
    "AVAX-USD": "avalanche-2",
    "MATIC-USD": "matic-network",
    "GALA-USD": "gala"
}

# عرض السعر الحالي
coin_id = symbol_map.get(ticker.upper())
if coin_id:
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
        response = requests.get(url)
        if response.status_code == 200:
            price = response.json()[coin_id]['usd']
            st.metric(label="💰 السعر الحالي", value=f"${price:,.4f}")
        else:
            st.warning("⚠️ تعذر جلب السعر الحالي من CoinGecko.")
    except:
        st.warning("⚠️ حدث خطأ أثناء الاتصال بـ CoinGecko.")
else:
    st.info("🔎 لا يمكن عرض السعر المباشر لهذه العملة حالياً.")
