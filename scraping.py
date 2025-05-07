import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import string
import re
import socket
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from core.config import settings
# Init this socket to avoid DNS resolution errors
def es_url_que_responde(url, timeout=5):  
    try:
        respuesta = requests.head(url, allow_redirects=True, timeout=timeout)

        if respuesta.status_code < 400:
            return extract_features_from_url(url)
        else:
            return False

    except requests.RequestException:
        return False


# buscar palabras clave en el texto
def keyword_detect(text, keywords):
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)

# numero de redirecciones
def count_url_redirects(url):
    try:
        response = requests.get(url, allow_redirects=True, timeout=10)
        return len(response.history)  # Cada objeto en .history representa una redirección
    except requests.RequestException:
        return -1  # O algún valor que indique error
    
def count_self_redirects(url):
    try:
        response = requests.get(url, allow_redirects=True, timeout=10)
        original_netloc = urlparse(url).netloc.lower().replace("www.", "")
        redirects = [
            r for r in response.history
            if urlparse(r.url).netloc.lower().replace("www.", "") == original_netloc
        ]
        return len(redirects)
    except requests.RequestException:
        return 0  # En caso de error en la petición

def extract_features_from_url(url: str) -> dict:
    #bank_keywords = ["bank", "transfer", "iban", "swift", "account", "balance", "loan", "credit"]
    #pay_keywords = ["payment", "checkout", "invoice", "paypal", "card", "stripe", "pay"]
    #crypto_keywords = ["bitcoin", "crypto", "ethereum", "blockchain", "wallet", "btc"]
    #social_domains = ["facebook.com", "twitter.com", "instagram.com", "linkedin.com"]
    try:
        # Descargar el HTML
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, timeout=10, headers=headers)
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')

        parsed = urlparse(url)
        domain = parsed.netloc
        path = parsed.path
        filename = domain.replace(".", "_") + ".txt"

        # Comprobaciones y conteos básicos
        is_https = 1 if parsed.scheme == "https" else 0
        domain_ip = 1 if re.match(r"\d+\.\d+\.\d+\.\d+", domain) else 0
        tld = domain.split('.')[-1]
        tld_len = len(tld)

        url_length = len(url)
        domain_length = len(domain)

        no_of_digits = sum(c.isdigit() for c in url)
        digit_ratio = no_of_digits / url_length

        no_of_letters = sum(c.isalpha() for c in url)
        letter_ratio = no_of_letters / url_length

        other_special_chars = re.findall(r'[=|?|&|@|#|%|$|:|;|[^a-zA-Z0-9]]', url)
        spatial_char_ratio = len(other_special_chars) / url_length

        has_favicon = 1 if soup.find("link", rel=lambda val: val and "icon" in val.lower()) else 0
        has_title = 1 if soup.title else 0
        title = soup.title.string.strip() if has_title else ""

        num_lines = len(html.splitlines())
        largest_line = max((len(line) for line in html.splitlines()), default=0)

        has_description = 1 if soup.find("meta", attrs={"name": "description"}) else 0

        has_submit = 1 if soup.find("input", {"type": "submit"}) else 0
        has_password = 1 if soup.find("input", {"type": "password"}) else 0
        has_hidden = 1 if soup.find("input", {"type": "hidden"}) else 0

        has_social = 1 if any(domain in html.lower() for domain in settings.social_domains) else 0
        has_external_form = 1 if any("http" in (form.get("action") or "") and domain not in form.get("action") for form in soup.find_all("form")) else 0

        num_img = len(soup.find_all("img"))
        num_css = len(soup.find_all("link", rel="stylesheet"))
        num_js = len(soup.find_all("script"))
        num_self_ref = sum(1 for a in soup.find_all("a", href=True) if domain in a["href"])
        num_empty_ref = sum(1 for a in soup.find_all("a", href=True) if a["href"].strip() == "#")
        num_external_ref = sum(1 for a in soup.find_all("a", href=True) if a["href"].startswith("http") and domain not in a["href"])
        # robots
        try:
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            robots_response = requests.get(robots_url, timeout=5, headers=headers)
            has_robots = 1 if robots_response.status_code == 200 else 0
        except:
            has_robots = 0
        # Unir todo el contenido textual
        text_to_check = f"{title} {soup.get_text()}"
        bank = 1 if keyword_detect(text_to_check, settings.bank_keywords) else 0
        pay = 1 if keyword_detect(text_to_check, settings.pay_keywords) else 0
        crypto = 1 if keyword_detect(text_to_check, settings.crypto_keywords) else 0
        pattern = r"%[0-9a-fA-F]{2}|\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}"
        matches = re.findall(pattern, url)
        def tokenize(text):
            return set(re.findall(r'\w+', text.lower()))

        url_tokens = tokenize(url)
        title_tokens = tokenize(title)

        URLTitleMatchScore = round(
            (len(url_tokens & title_tokens) / len(title_tokens) * 100) if title_tokens else 0,3
        )
        # CharContinuationRate
        def char_continuation_rate(u):
            if len(u) <= 1:
                return 1.0
            same = sum(1 for i in range(1, len(u)) if u[i] == u[i - 1])
            return round(same / len(u), 3)
        #TLDLegitimateProb
        def tld_legitimate_prob(tld):
            #comunes = {'com', 'org', 'net', 'edu', 'gov'}
            #sospechosos = {'xyz', 'top', 'gq', 'tk', 'ml'}
            if tld in settings.legitimate_tlds:
                return 0.9
            elif tld in settings.suspect_tlds:
                return 0.1
            else:
                return 0.5
        # URLCharProb
        def url_char_prob(u):
            normales = string.ascii_letters + string.digits + "-_.~/:%"
            normales_count = sum(1 for c in u if c in normales)
            return round(normales_count / len(u), 3) if u else 0
        
        # mapping of TLD to numeric values
        tld_mapping = joblib.load(settings.model_map_tld)
        mapping_tld = tld_mapping.get(tld, 0)  # Default to 0 if TLD not found

        # Campos estadísticos / heurísticos estimados
        features = {
            #"FILENAME": filename,
            #"URL": url,
            "URLLength": url_length,
            #"Domain": domain,
            "DomainLength": domain_length,
            "IsDomainIP": domain_ip,
            "TLD": mapping_tld, #es un numero necesario mapeado
            "URLSimilarityIndex": 100,  # Asumido 100% consigo mismo
            "CharContinuationRate": char_continuation_rate(url), 
            "TLDLegitimateProb": tld_legitimate_prob(tld),  
            "URLCharProb": url_char_prob(url), 
            "TLDLength": tld_len,
            "NoOfSubDomain": domain.count('.') - 1,
            "HasObfuscation": 1 if matches else 0,
            "NoOfObfuscatedChar": len(matches),
            "ObfuscationRatio": round(len(matches) / url_length, 3) if url_length > 0 else 0.0,
            "NoOfLettersInURL": no_of_letters,
            "LetterRatioInURL": round(letter_ratio, 3),
            "NoOfDegitsInURL": no_of_digits,
            "DegitRatioInURL": round(digit_ratio, 3),
            "NoOfEqualsInURL": url.count('='),
            "NoOfQMarkInURL": url.count('?'),
            "NoOfAmpersandInURL": url.count('&'),
            "NoOfOtherSpecialCharsInURL": len(other_special_chars),
            "SpacialCharRatioInURL": round(spatial_char_ratio, 3),
            "IsHTTPS": is_https,
            "LineOfCode": num_lines,
            "LargestLineLength": largest_line,
            "HasTitle": has_title,
            #"Title": title,
            "DomainTitleMatchScore": 1 if title and domain.lower() in title.lower() else 0,
            "URLTitleMatchScore": URLTitleMatchScore,
            "HasFavicon": has_favicon,
            "Robots": has_robots,
            "IsResponsive": 0,  # Requiere renderizado JS
            "NoOfURLRedirect": count_url_redirects(url), 
            "NoOfSelfRedirect": count_self_redirects(url),
            "HasDescription": has_description,
            "NoOfPopup": 0,  # Requiere JS/render
            "NoOfiFrame": len(soup.find_all("iframe")),
            "HasExternalFormSubmit": has_external_form,
            "HasSocialNet": has_social,
            "HasSubmitButton": has_submit,
            "HasHiddenFields": has_hidden,
            "HasPasswordField": has_password,
            "Bank": bank,
            "Pay": pay,
            "Crypto": crypto,
            "HasCopyrightInfo": 1 if "©" in html else 0,
            "NoOfImage": num_img,
            "NoOfCSS": num_css,
            "NoOfJS": num_js,
            "NoOfSelfRef": num_self_ref,
            "NoOfEmptyRef": num_empty_ref,
            "NoOfExternalRef": num_external_ref
            # 'label' no se incluye ya que es objetivo
        }
        input_df = pd.DataFrame([features])
        model = joblib.load(settings.model_path)
        prediction = model.predict(input_df)[0]
        print(f"Predicción: {prediction}")
        return int(prediction)

    except Exception as e:
        print(f"Error al procesar la URL: {e}")
        return {}