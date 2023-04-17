from re import compile

compileRegexs = {
    'punctuationRegex': compile(r'[^0-9a-zA-Z_]'),
    'specialCharactersRegex': compile(r'[!\?\.\,\$\-\_)(]'),
    'parenthesisRemoveRegex': compile(r'\([^()]*\)'),
    'blankSpacesRegex': compile(r'\s{2,}'),
    'smallWordsRegex': compile(r'\W*\b\w{1,2}\b'),
    'zipCodeRegex': compile(r'\d{5}(\-)?\s?\d{3}'),
    'addressRegex': compile(r'(RUA|Rua|R.|AVENIDA|Avenida|AV.|TRAVESSA|Travessa|TRAV.|Trav.)([a-zA-Z_\s]+)[,\s\-]+(\d+)\s?([-/\da-zDA-Z\\ ]+)[,\s\-]\s?([a-zA-Z_\s]+)'),
    'linkRegex': compile(r'http\S+'),
    'dateRegex': compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b|\b\d{4}[/-]\d{2}[/-]\d{2}\b|\b\d{1,2}\s\w{3,}\s\d{4}\b|\b\d{1,2}[.]\d{1,2}[.]\d{4}\b'),
    'telephoneRegex': compile(r'\+?\d{2}?\s?\(?\d{2}\)?\s?\d{4,5}-?\d{4}\b|\(?\d{2}\)?\s?\d{4}-?\d{4}\b'),
    'emailRegex': compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    'emojiRegex': compile(r':[^:\s]*(?:::[^:\s]*)*:'),
    'numericRegex': compile(r'\d+')
}
