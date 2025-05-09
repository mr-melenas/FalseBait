class Settings:
    def __init__(self):
        self.model = "RandomForestClassifier"  # Model name
        self.model_type = "Classification"  # Model type
        self.proyect_name = "Classification of URL"
        self.version = "0.0.1"
        self.description = "Categorization of URLs using machine learning for phishing detection"
        self.api_prefix = "/api" # API prefix
        self.api_version = "/v1" # API version
        self.model_path_A = "data/model_clf_A.pkl" # Path to the model file A
        self.model_path_B = "data/model_clf_B.pkl" # Path to the model file B
        self.test_data_logs = "csv/test_data_logs.csv" # Path to the test data logs
        self.combined_data = "csv/combine_model_sql.csv" # Path to the combined data
        self.model_path = "data/model_clf.pkl" # Path to the model file
        self.model_map_tld = "data/tld_mapping.pkl" # Mapping for TLDs
        self.model_features = [
            "url",
            "is_https",
            "domain_ip",
            "tld",
            "tld_len",
            "url_length",
            "domain_length",
            "no_of_digits",
            "digit_ratio",
            "no_of_letters",
            "letter_ratio",
            "spatial_char_ratio",
            "has_favicon",
            "has_title",
            "title",
            "num_lines",
            "largest_line",
            "has_description",
            "has_submit",
            "has_password",
            "has_hidden",
            "has_social",
            "has_external_form",
            "num_img",
            "num_css",
            "num_js",
            "num_self_ref",
            "num_empty_ref",
            "num_external_ref"
        ]
        self.social_domains = ["facebook.com", "twitter.com", "instagram.com", "linkedin.com", "youtube.com", "pinterest.com", "snapchat.com", "tiktok.com", "whatsapp.com", "reddit.com"]
        self.crypto_keywords = ["bitcoin", "crypto", "ethereum", "blockchain", "wallet", "btc"]
        self.pay_keywords = ["payment", "checkout", "invoice", "paypal", "card", "stripe", "pay"]
        self.bank_keywords = ["bank", "transfer", "iban", "swift", "account", "balance", "loan", "credit"]
        self.suspect_tlds = [
                        "tk",  # Tokelau - gratuito, abusado frecuentemente
                        "ml",  # Mali
                        "ga",  # Gabón
                        "cf",  # República Centroafricana
                        "gq",  # Guinea Ecuatorial
                        "xyz", # Muy barato y popular entre spammers
                        "top", # Similar a xyz
                        "club", # También común por bajo coste
                        "work",
                        "click",
                        "support",
                        "loan",
                        "win",
                        "men",
                        "stream",
                        "download",
                        "review",
                        "date",
                        "faith",
                        "trade",
                        "science",
                        "party",
                        "cam",
                        "host",
                        "space",
                        "biz",   # Usado para sitios falsos de empresas
                        "info",  # También usado como alternativa para engañar
                        "pw",    # Palau - asociado a muchos dominios falsos
                        "buzz",  # En algunos reportes aparece con actividad sospechosa
                    ]
        self.legitimate_tlds = [
                        "com",   # Comercial (el más usado del mundo)
                        "org",   # Organizaciones no lucrativas
                        "net",   # Infraestructura de red
                        "edu",   # Instituciones educativas (EE. UU.)
                        "gov",   # Gobierno de EE. UU.
                        "mil",   # Militar (EE. UU.)
                        "int",   # Organismos internacionales

                        # Europeos comunes
                        "es",    # España
                        "fr",    # Francia
                        "de",    # Alemania
                        "it",    # Italia
                        "uk",    # Reino Unido
                        "nl",    # Países Bajos
                        "pt",    # Portugal

                        # Otros populares y confiables
                        "us",    # Estados Unidos
                        "ca",    # Canadá
                        "au",    # Australia
                        "ch",    # Suiza
                        "jp",    # Japón
                        "kr",    # Corea del Sur
                        "se",    # Suecia
                        "no",    # Noruega
                        "fi",    # Finlandia
                        "br",    # Brasil
                        "mx",    # México
                        "in",    # India
                        "cn",    # China (aunque requiere más control)

                        # Nuevos gTLD de empresas conocidas
                        "google",
                        "microsoft",
                        "ibm",
                        "apple"
                    ]
settings = Settings()
