import nltk
import emoji
from re import sub
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from unicodedata import normalize
from nltk.tokenize import word_tokenize
from modules.nlp.pre_processing.assets.compileRegexs import compileRegexs


nltk.download(['stopwords'])

STOPWORDS = stopwords.words('portuguese')
STOPWORDS.remove('sem')
STOPWORDS.remove('não')
STOPWORDS.remove('há')


def cleaning(text: str) -> str:
    """
        Aplica a limpeza na coluna de texto
    """
    cleanedText = str(text).lower()
    cleanedText = replaceTag(cleanedText)
    cleanedText = removeHtml(cleanedText)
    cleanedText = removeStopwords(cleanedText)
    cleanedText = removeEmojis(cleanedText)
    cleanedText = normalizeText(cleanedText)
    cleanedText = removePunctuation(cleanedText)
    cleanedText = replaceBlanks(cleanedText)
    cleanedText = clearText(cleanedText)
    return cleanedText


def prepareData(text: str) -> str:
    """
        Aplica um tratamento nas outras colunas da base
    """
    cleanedText = str(text).lower()
    cleanedText = removeHtml(cleanedText)
    cleanedText = removeStopwords(cleanedText)
    cleanedText = removeEmojis(cleanedText)
    cleanedText = normalizeText(cleanedText)
    cleanedText = sub('[\(\)]', ' ', cleanedText)
    cleanedText = sub('[-_\"\']', ' ', cleanedText)
    cleanedText = replaceBlanks(cleanedText)
    cleanedText = clearText(cleanedText)
    return cleanedText


def replaceTag(text: str) -> str:
    """
        Realiza a substituição de alguns padrões recorrentes no texto
    """
    replacedText = compileRegexs['linkRegex'].sub(' linkTag ', text)
    replacedText = compileRegexs['emailRegex'].sub(' emailTag ', replacedText)
    replacedText = compileRegexs['telephoneRegex'].sub(' telefoneTag ', replacedText)
    replacedText = compileRegexs['addressRegex'].sub(' enderecoTag ', replacedText)
    replacedText = compileRegexs['dateRegex'].sub(' dataTag ', replacedText)
    replacedText = compileRegexs['numericRegex'].sub(' decimalTag ', replacedText)
    return replacedText


def removeHtml(text:str) -> str:
    """
        Remove tags de html
    """
    text = text.replace('<br/>', ' ')
    soup = BeautifulSoup(text,'html5lib')
    text = soup.get_text(strip = True)    
    return text


def removeStopwords(text: str) -> str:
    """
        Realiza a remoção de palavras irrelevantes do texto
    """
    return ' '.join(word for word in text.split() if word not in STOPWORDS)


def removeEmojis(text: str) -> str:
    """
        Remove caracteres de emojis do texto
    """
    text = emoji.demojize(text)
    text = compileRegexs['emojiRegex'].sub(' ', str(text))
    return text
    

def normalizeText(text: str) -> str:
    """
        Realiza a normalização dos textos retirando a acentuação
    """
    return normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')


def removePunctuation(text: str) -> str:
    """
        Realiza a remoção da pontuação dos textos
    """
    cleanedText = compileRegexs['punctuationRegex'].sub(' ', text)
    cleanedText = compileRegexs['specialCharactersRegex'].sub(' ', cleanedText)
    return cleanedText


def replaceBlanks(text: str) -> str:
    """
        Realiza as remoção de marcações de quebra de linha, 'carriage return' e espaçamento dos textos
    """
    text = text.replace('\r', ' ')
    text = text.replace('\n', ' ')
    return text


def clearText(text: str) -> str:
    """
        Realiza a remoção de espaços em brancos e palavras muito pequenas
    """
    text = compileRegexs['smallWordsRegex'].sub(' ', str(text))
    text = compileRegexs['blankSpacesRegex'].sub(' ', str(text))
    text = text.strip()
    return text

