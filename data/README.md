# Data / Dados

## PT — Português

Os dados brutos utilizados neste estudo **não estão incluídos** neste repositório por razões de confidencialidade institucional. Os dados pertencem ao LIMS (Laboratory Information Management System) da CAER — Companhia de Águas e Esgotos de Roraima.

### Como obter dados equivalentes

Para testar o pipeline com dados próprios, o arquivo CSV deve ser colocado nesta pasta com o nome:

```
data/dados-analise-agua.csv
```

**Separador:** `;` (ponto e vírgula)

**Colunas necessárias:**

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `nome_cidade` | string | Cidade (filtro: `BOA VISTA`) |
| `nome_manancial` | string | Manancial (filtro: `RIO BRANCO`) |
| `nome_sistema_abastacimento` | string | Sistema (filtro: `SÃO PEDRO`) |
| `data_coleta` | date | Data de coleta |
| `ph` | float | pH da água |
| `turbidez` | float | Turbidez (NTU) |
| `cor` | float | Cor aparente (uH) |
| `sulfato` | float | Sulfato — proxy da dosagem de coagulante (mg/L) |

---

## EN — English

Raw data used in this study is **not included** due to institutional confidentiality. Data belongs to the LIMS of CAER (Water and Sanitation Company of Roraima, Brazil).

To test the pipeline with your own data, place a `;`-separated CSV file at `data/dados-analise-agua.csv` with the columns described above.
