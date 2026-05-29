# multilang

Dictionary-based text translator with persistent cache for multi-language notebooks.

## Installation

```bash
pip install deep-translator
pip install -e "D:\Cursos\DataCamp\Data Scientist with Python\Projects\ds_toolkit\multilang"
```

## Quick Start

```python
from multilang import LangMap

# EN notebook
lang = LangMap(source="pt", target="en")

# PT notebook — passthrough, zero API calls
lang = LangMap(source="pt", target="pt")

# Usage is identical in both notebooks
labels = lang({
    "title":    "Distribuição de Churn por Contrato",
    "xlabel":   "Tipo de Contrato",
    "ylabel":   "Densidade",
    "subtitle": "Clientes ativos — 2024",
})

ax.set_title(labels["title"])
ax.set_xlabel(labels["xlabel"])
ax.set_ylabel(labels["ylabel"])
```

## Cache

Translations are persisted in `multilang/cache/{source}_{target}.json`. Only texts **absent** from the cache are sent to the API — repeated calls are instant.

```python
# Inspect the current cache
lang.show_cache()

# Pre-populate cache before running a notebook
lang.warm_up(["Distribuição", "Tipo de Contrato", "Densidade"])

# Force re-translation (ignores cache)
lang({"title": "Distribuição"}, force=True)
```

## Offline Mode

```python
lang = LangMap(source="pt", target="en", offline=True)
# Raises RuntimeError if a text is not already cached
```

## Switch Language

```python
lang = LangMap(source="pt", target="en")
lang.switch(target="es")   # now translates to Spanish, loads es cache
```

## Supported Languages

`deep-translator` supports all languages available via Google Translate. Common codes:

| Code | Language   |
|------|------------|
| pt   | Portuguese |
| en   | English    |
| es   | Spanish    |
| fr   | French     |
| de   | German     |
| it   | Italian    |
| ja   | Japanese   |
| zh-CN| Chinese    |

Full list: `from deep_translator import GoogleTranslator; GoogleTranslator.get_supported_languages(as_dict=True)`
