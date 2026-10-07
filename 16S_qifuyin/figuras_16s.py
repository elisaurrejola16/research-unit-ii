"""
Lectores de los exportes de QIIME2 para el notebook de figuras.

Cada funcion resuelve una trampa concreta del formato de salida de QIIME2.
Lo unico que hacen es devolver DataFrames limpios: no dibujan nada.

Uso:
    import figuras_16s as f16
    tabla = f16.leer_tabla_asv("qiime2_export/asv_export/asv_table.tsv")
"""

import numpy as np
import pandas as pd

RANGOS = ["dominio", "phylum", "clase", "orden", "familia", "genero"]


# --- lectores -----------------------------------------------------------------

def leer_tabla_asv(path):
    """Tabla de conteos de `biom convert --to-tsv`.

    TRAMPA: la primera linea es el comentario '# Constructed from biom file'.
    Sin skiprows=1 el header queda corrido una fila.
    Devuelve: ASVs en filas, muestras en columnas, conteos CRUDOS.
    """
    t = pd.read_csv(path, sep="\t", skiprows=1, index_col=0)
    t.index.name = "ASV_ID"
    return t


def leer_taxonomia(path, incertae_a_na=True):
    """taxonomy.tsv exportado de taxonomy.qza, con el linaje partido en columnas.

    TRAMPA 1: tiene TODOS los ASVs de DADA2, no solo los de la tabla filtrada.
              Siempre unir DESDE la tabla (how='right'/'left' con la tabla a la
              izquierda), nunca al reves, o reaparecen mitocondrias y cloroplastos.
    TRAMPA 2: 'Incertae_Sedis' no es un genero resuelto. Con incertae_a_na=True
              pasa a NaN, que es lo correcto para contar cobertura taxonomica.
    """
    tax = pd.read_csv(path, sep="\t", index_col=0)
    tax.index.name = "ASV_ID"
    sp = tax["Taxon"].str.split(";", expand=True).reindex(columns=range(len(RANGOS)))
    sp.columns = RANGOS
    for c in RANGOS:
        sp[c] = (sp[c].str.strip()
                      .str.replace(r"^[a-z]__", "", regex=True)
                      .replace({"": None}))
    if incertae_a_na:
        sp.loc[sp["genero"] == "Incertae_Sedis", "genero"] = None
    return pd.concat([sp, tax[["Taxon", "Confidence"]]], axis=1)


def leer_metadata(path, orden_grupo=("Con", "Mod", "QFY")):
    """metadata.tsv de QIIME2.

    TRAMPA: la segunda fila es '#q2:types', que no es una muestra. Si no se saca,
            aparece como una barra fantasma en todos los graficos.
    Deja `grupo` como Categorical ordenado, asi el orden Con/Mod/QFY se respeta
    solo en todos los ejes, groupby y leyendas.
    """
    md = pd.read_csv(path, sep="\t", index_col=0)
    md = md[~md.index.astype(str).str.startswith("#")]
    if "grupo" in md.columns:
        md["grupo"] = pd.Categorical(md["grupo"], categories=list(orden_grupo), ordered=True)
    return md


def leer_alfa(path, nombre=None):
    """alpha-diversity.tsv de un *_vector.qza exportado. Devuelve una Series."""
    a = pd.read_csv(path, sep="\t", index_col=0)
    s = a.iloc[:, 0]
    s.name = nombre or s.name
    return s


def leer_matriz_distancia(path):
    """distance-matrix.tsv exportado de un *_distance_matrix.qza. DataFrame cuadrado."""
    d = pd.read_csv(path, sep="\t", index_col=0)
    d.columns = d.columns.astype(str)
    assert d.shape[0] == d.shape[1], f"la matriz no es cuadrada: {d.shape}"
    assert list(d.index) == list(d.columns), "filas y columnas en distinto orden"
    return d


def leer_ordinacion(path):
    """ordination.txt de un *_pcoa_results.qza exportado.

    Es el MISMO PCoA que dibuja Emperor, asi que los ejes y los porcentajes del
    grafico de matplotlib van a coincidir exactamente con lo que ya se miro en
    el .qzv. No recalcula nada.

    Devuelve (coords, prop_explicada):
        coords        DataFrame muestras x ejes, columnas 'PC1', 'PC2', ...
        prop_explicada Series indexada igual, en FRACCION (0-1), no en %.
    """
    with open(path) as fh:
        lineas = fh.read().split("\n")

    prop, coords, ids = None, [], []
    i = 0
    while i < len(lineas):
        ln = lineas[i]
        if ln.startswith("Proportion explained"):
            prop = np.array([float(x) for x in lineas[i + 1].split("\t")])
            i += 2
            continue
        if ln.startswith("Site\t"):
            n = int(ln.split("\t")[1])
            for fila in lineas[i + 1: i + 1 + n]:
                partes = fila.split("\t")
                ids.append(partes[0])
                coords.append([float(x) for x in partes[1:]])
            i += 1 + n
            continue
        i += 1

    assert prop is not None and ids, f"no pude parsear {path}"
    ejes = [f"PC{k + 1}" for k in range(len(coords[0]))]
    return (pd.DataFrame(coords, index=ids, columns=ejes),
            pd.Series(prop, index=ejes, name="prop_explicada"))


# --- transformaciones ---------------------------------------------------------

def abundancia_relativa(tabla, tax, nivel="familia", sin_asignar="sin asignar"):
    """Colapsa la tabla de ASVs a un nivel taxonomico, en abundancia relativa.

    ORDEN DE OPERACIONES (esto es lo importante): relativiza POR MUESTRA y
    despues colapsa. Al reves --- sumar conteos crudos y relativizar al final ---
    las muestras mas profundas dominan el resultado. Es un error que se publica.

    Devuelve: taxones en filas, muestras en columnas, fracciones que suman 1.
    """
    rel = tabla / tabla.sum(axis=0)
    etiqueta = tax.reindex(tabla.index)[nivel].fillna(sin_asignar)
    return rel.groupby(etiqueta).sum()


def media_por_grupo(rel, md, col="grupo"):
    """Promedia abundancias relativas por grupo. Taxones en filas, grupos en columnas."""
    m = rel.T.join(md[col]).groupby(col, observed=True).mean().T
    return m


def top_con_otros(m, n=10, etiqueta="Otros"):
    """Deja los n taxones mas abundantes (por maximo entre grupos) y suma el resto.

    'Otros' va SIEMPRE al final, para que en las barras apiladas quede arriba y
    no se confunda con un taxon real.
    """
    orden = m.max(axis=1).sort_values(ascending=False).index
    top = m.loc[orden[:n]]
    resto = m.loc[orden[n:]].sum(axis=0)
    if resto.any():
        top = pd.concat([top, resto.to_frame(etiqueta).T])
    return top


def lecturas_equivalentes(pct, lecturas_por_muestra):
    """Convierte un % de abundancia relativa a lecturas por muestra.

    El filtro anti-ruido: bajo ~10 lecturas, una diferencia entre grupos es azar
    de muestreo y no se reporta.
    """
    return pct / 100 * lecturas_por_muestra

    # --- ANCOM-BC2 (P9) -----------------------------------------------------------

_CANT = ["lfc", "se", "W", "p", "q", "diff", "diff_robust", "passed_ss"]


def _leer_jsonl(path):
    """Un .jsonl de ANCOM-BC2 -> (DataFrame indexado por taxon, dict de campos).

    TRAMPA: la PRIMERA linea no es un dato, es el encabezado (`doctype`) con la
            lista de campos. Leerla como fila mete una fila fantasma de NaN.
            En `fields[*]['extra']` viene de que variable y nivel es cada columna,
            y cual es la referencia del contraste.
    """
    import json
    filas = []
    with open(path) as fh:
        cab = json.loads(fh.readline())
        assert "doctype" in cab, f"{path}: la primera linea no es el encabezado"
        campos = {c["name"]: c.get("extra", {}) for c in cab["fields"]}
        for linea in fh:
            if linea.strip():
                filas.append(json.loads(linea))
    df = pd.DataFrame(filas).set_index("taxon")
    return df, campos


def leer_ancombc2(carpeta, variable="grupo"):
    """Carpeta de `qiime tools export` de un ANCOMBC2Output -> tabla ordenada.

    Devuelve una fila por (taxon, contraste) con lfc, se, W, p, q y las banderas
    del analisis de sensibilidad. Una sola tabla, lista para filtrar y para el
    suplementario.

    `variable='grupo'` deja solo los contrastes de grupo y descarta `(Intercept)`
    y las covariables (sexo). Con variable=None devuelve todo.

    OJO: el `lfc` esta en escala logaritmica, no es un fold-change.
    """
    import os
    partes = []
    for cant in _CANT:
        path = os.path.join(carpeta, f"{cant}.jsonl")
        if not os.path.exists(path):
            continue
        df, campos = _leer_jsonl(path)
        largo = df.stack().rename(cant)
        largo.index.names = ["taxon", "columna"]
        partes.append(largo)
    assert partes, f"{carpeta}: no encontre ningun .jsonl de ANCOM-BC2"

    t = pd.concat(partes, axis=1).reset_index()
    meta = pd.DataFrame([{"columna": k,
                          "variable": v.get("variable"),
                          "nivel": v.get("level"),
                          "referencia": v.get("reference")}
                         for k, v in campos.items() if k != "taxon"])
    t = t.merge(meta, on="columna", how="left")
    if variable is not None:
        t = t[t["variable"] == variable].copy()
        assert len(t), f"no hay columnas de la variable '{variable}' en {carpeta}"
    t["contraste"] = t["nivel"] + " vs " + t["referencia"]
    # OJO: reset ANTES de partir el linaje. Si se filtro por variable, el indice
    # quedo con huecos y el concat de abajo desalinea e inventa filas con NaN.
    t = t.reset_index(drop=True)

    # el linaje se parte igual que en leer_taxonomia, para que los nombres calcen
    sp = t["taxon"].str.split(";", expand=True).reindex(columns=range(len(RANGOS)))
    sp.columns = RANGOS
    # OJO: si el linaje se corta en familia, las columnas que faltan quedan todas
    # NaN y pandas las tipa float -> .str explota. astype(object) lo arregla.
    sp = sp.astype("object")
    for c in RANGOS:
        sp[c] = (sp[c].str.strip()
                      .str.replace(r"^[a-z]__", "", regex=True)
                      .replace({"": None}))
    t = pd.concat([t, sp[["phylum", "familia"]]], axis=1)

    cols = ["familia", "contraste", "lfc", "se", "W", "p", "q",
            "diff", "diff_robust", "passed_ss", "phylum", "taxon"]
    t = t[[c for c in cols if c in t.columns]]
    return t.sort_values(["contraste", "q"]).reset_index(drop=True)


def leer_ceros_estructurales(carpeta):
    """structural-zeros.jsonl -> DataFrame booleano, un taxon por fila.

    Una familia marcada True en un grupo esta ausente de verdad en ese grupo, no
    ausente por poca profundidad. Es un resultado mas fuerte que cualquier lfc y
    se reporta aparte, no con q.
    """
    import os
    df, campos = _leer_jsonl(os.path.join(carpeta, "structural-zeros.jsonl"))
    df.columns = [campos[c].get("level", c) for c in df.columns]
    sp = (pd.Series(df.index).str.split(";", expand=True)
            .reindex(columns=range(len(RANGOS))).astype("object"))
    df.insert(0, "familia", (sp[4].str.strip()
                                 .str.replace(r"^[a-z]__", "", regex=True)
                                 .replace({"": None})).values)
    return df
    

# --- ANCOM-BC2 de rutas, PICRUSt2 (P10) ---------------------------------------

def _leer_jsonl_id(path):
    """Como _leer_jsonl pero sin asumir que la columna de id se llama "taxon".

    TRAMPA: la salida de rutas usa el mismo formato que la de familias, pero el
            identificador puede venir con otro nombre de columna. Se toma el
            PRIMER campo del encabezado, que siempre es el id.
    """
    import json
    filas = []
    with open(path) as fh:
        cab = json.loads(fh.readline())
        assert "doctype" in cab, f"{path}: la primera linea no es el encabezado"
        campos = {c["name"]: c.get("extra", {}) for c in cab["fields"]}
        id_col = cab["fields"][0]["name"]
        for linea in fh:
            if linea.strip():
                filas.append(json.loads(linea))
    return pd.DataFrame(filas).set_index(id_col), campos, id_col


def leer_ancombc2_rutas(carpeta, variable="grupo"):
    """Carpeta exportada del ANCOMBC2Output de RUTAS (P10) -> tabla ordenada.

    Una fila por (ruta, contraste) con lfc, se, W, p, q y las banderas del
    analisis de sensibilidad. Es el equivalente de leer_ancombc2 para PICRUSt2:
    misma estructura de archivos, pero el identificador es una ruta MetaCyc
    (`PWY-6151`) y NO se parsea como linaje.

    OJO: el `lfc` esta en escala logaritmica, no es un fold-change.
    OJO: son rutas PREDICHAS. Cualquier resultado de aca es una hipotesis
         funcional, no una funcion medida.
    """
    import os
    partes = []
    for cant in _CANT:
        path = os.path.join(carpeta, f"{cant}.jsonl")
        if not os.path.exists(path):
            continue
        df, campos, id_col = _leer_jsonl_id(path)
        largo = df.stack().rename(cant)
        largo.index.names = ["ruta", "columna"]
        partes.append(largo)
    assert partes, f"{carpeta}: no encontre ningun .jsonl de ANCOM-BC2"

    t = pd.concat(partes, axis=1).reset_index()
    meta = pd.DataFrame([{"columna": k,
                          "variable": v.get("variable"),
                          "nivel": v.get("level"),
                          "referencia": v.get("reference")}
                         for k, v in campos.items() if k != id_col])
    t = t.merge(meta, on="columna", how="left")
    if variable is not None:
        t = t[t["variable"] == variable].copy()
        assert len(t), f"no hay columnas de la variable '{variable}' en {carpeta}"
    t["contraste"] = t["nivel"] + " vs " + t["referencia"]
    t = t.reset_index(drop=True)

    cols = ["ruta", "contraste", "lfc", "se", "W", "p", "q",
            "diff", "diff_robust", "passed_ss"]
    t = t[[c for c in cols if c in t.columns]]
    return t.sort_values(["contraste", "q"]).reset_index(drop=True)


def leer_rutas_abundancia(path):
    """path_abun_unstrat_descrip.tsv.gz -> (abundancias rutas x muestras, descripciones).

    La primera columna es el id de la ruta, la segunda la descripcion legible y
    el resto son las muestras. Devuelve las dos cosas por separado.

    OJO: los valores NO son conteos: son sumas ponderadas por el numero de
         copias del 16S, y por eso tienen decimales.
    """
    t = pd.read_csv(path, sep="\t", index_col=0)
    desc = t.pop("description") if "description" in t.columns else None
    return t.astype(float), desc


def clr_muestras(t, pseudo=1.0):
    """CLR por muestra (columnas = muestras). Devuelve muestras x rutas.

    El CLR es la transformacion estandar para datos composicionales: deja las
    distancias comparables entre muestras de distinta profundidad.

    OJO: el pseudoconteo es necesario porque el log de 0 no existe, y la
         eleccion del valor afecta a las rutas con muchos ceros.
    """
    import numpy as np
    x = np.log(t.T + pseudo)
    return x.sub(x.mean(axis=1), axis=0)