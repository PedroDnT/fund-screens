import logging

import pandas as pd
from pathlib import Path
from typing import Any
from config.settings import Config
from src.utils.cnpj_utils import normalize_cnpj_list, normalize_cnpj_series

logger = logging.getLogger(__name__)


class DataProcessor:
    """Processador de dados da CVM (leitura, limpeza, transformação)"""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Padroniza nomes de colunas removendo espaços e BOM."""
        df = df.copy()
        df.columns = (
            df.columns.astype(str)
            .str.replace('\ufeff', '', regex=False)
            .str.replace('ï»¿', '', regex=False)
            .str.replace('\xa0', '', regex=False)
            .str.strip()
            .str.upper()
        )
        return df

    @staticmethod
    def _apply_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
        """Aplica aliases conhecidos para nomes de colunas padrão."""
        df = df.copy()
        aliases = {
            'CNPJ_FUNDO': ['CNPJ_FUNDO_CLASSE'],
            'DT_COMPTC': ['DT_COMPTC_CLASSE'],
        }

        for target, alternatives in aliases.items():
            if target in df.columns:
                continue
            for alt in alternatives:
                if alt in df.columns:
                    df[target] = df[alt]
                    break

        return df

    # Translation table for Brazilian number format (created once, reused)
    _BRAZILIAN_NUMBER_TRANS = str.maketrans({'.': '', ',': '.'})

    @staticmethod
    def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """Converte colunas numéricas tratando separadores brasileiros."""
        df = df.copy()
        for col in columns:
            if col not in df.columns:
                continue
            series = df[col]
            if pd.api.types.is_string_dtype(series) or series.dtype == object:
                # Use str.translate for faster simultaneous replacements
                cleaned = series.astype(str).str.translate(DataProcessor._BRAZILIAN_NUMBER_TRANS)
                df[col] = pd.to_numeric(cleaned, errors='coerce')
            else:
                df[col] = pd.to_numeric(series, errors='coerce')
        return df

    @staticmethod
    def _normalize_cnpj_series(series: pd.Series) -> pd.Series:
        """Normaliza CNPJ removendo pontuação e preservando NAs.

        Delegado a src.utils.cnpj_utils, a unica fonte de verdade para CNPJ.
        Valores que nao podem ser um CNPJ (vazios, sem digitos, com mais de 14
        digitos) viram pd.NA em vez de um identificador sintetico.

        Args:
            series: Série pandas contendo CNPJs em diversos formatos

        Returns:
            Série pandas com CNPJs normalizados (14 dígitos) ou NAs preservados
        """
        return normalize_cnpj_series(series)

    @classmethod
    def _normalize_cnpj_list(cls, cnpj_list: list[str]) -> list[str]:
        """Normaliza lista de CNPJs para comparação consistente."""
        return normalize_cnpj_list(cnpj_list)
    def _read_csv_generic(self,
                         file_path: Path,
                         file_type: str,
                         encoding: str = 'latin1',
                         sep: str = ';',
                         date_cols: list[str] | None = None,
                         cnpj_cols: list[str] | None = None,
                         numeric_cols: list[str] | None = None,
                         usecols: list[str] | None = None,
                         dtype: dict[str, Any] | None = None) -> pd.DataFrame:
        """
        Generic CSV reader with common transformations.

        Consolidates duplicate logic from read_informe_diario, read_cda, and read_cadastro.

        Args:
            file_path: Path to CSV file
            file_type: Type of file for logging (e.g., "Informe Diario", "CDA", "Cadastro")
            encoding: File encoding
            sep: CSV separator
            date_cols: List of date columns to parse
            cnpj_cols: List of CNPJ columns to normalize
            numeric_cols: List of numeric columns to coerce
            usecols: Restrict parsing to these columns. CVM monthly files run to
                hundreds of MB, so narrowing the read is the cheapest way to cut
                memory. Defaults to None (read everything), because downstream
                analyzers reach for columns beyond the ones normalized here.
            dtype: Explicit dtypes, to skip pandas' type inference on wide files

        Returns:
            Processed DataFrame or empty DataFrame on error
        """
        try:
            df = pd.read_csv(
                file_path, encoding=encoding, sep=sep, usecols=usecols, dtype=dtype,
                # Infer each column's type from the whole file rather than chunk
                # by chunk. Chunked inference can type one column two ways in one
                # file -- int64 where a chunk happens to hold only digits, object
                # where it does not -- and the halves then stringify differently
                # ('123' vs '123.0'), which silently breaks joins on code and
                # identifier columns.
                #
                # Reading a real CVM month raised DtypeWarning on four columns.
                # Comparing both settings across every file of that month found
                # no column whose dtype actually differed, so nothing was being
                # corrupted in practice -- this closes the hazard and quiets a
                # warning on every run, rather than fixing a live bug. The files
                # are tens of MB, so reading them whole costs nothing worth
                # having.
                low_memory=False,
            )

            # Padronizar nomes de colunas
            df = self._normalize_columns(df)
            df = self._apply_column_aliases(df)

            # Converter datas
            if date_cols:
                for col in date_cols:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], format='%Y-%m-%d', errors='coerce')

            # Normalizar CNPJs
            if cnpj_cols:
                for col in cnpj_cols:
                    if col in df.columns:
                        df[col] = self._normalize_cnpj_series(df[col])

            # Converter valores numéricos
            if numeric_cols:
                df = self._coerce_numeric(df, numeric_cols)

            return df
        except Exception as e:
            logger.error(f"Erro ao ler {file_type} {file_path}: {e}")
            return pd.DataFrame()

    def read_informe_diario(self, file_path: Path,
                           encoding: str = 'latin1',
                           sep: str = ';',
                           consolidate_subclasses: bool = True) -> pd.DataFrame:
        """Lê arquivo de Informe Diário.

        Since RCVM 175 the informe has one row per *subclass* per day for a class
        that has subclasses. By default those rows are folded into one row per
        class per day (``consolidate_subclasses``). Pass False for the raw rows,
        e.g. to read several monthly files and consolidate once after concat.
        """
        df = self._read_csv_generic(
            file_path=file_path,
            file_type="Informe Diario",
            encoding=encoding,
            sep=sep,
            date_cols=['DT_COMPTC'],
            cnpj_cols=['CNPJ_FUNDO'],
            numeric_cols=['VL_TOTAL', 'VL_QUOTA', 'VL_PATRIM_LIQ', 'CAPTC_DIA', 'RESG_DIA', 'NR_COTST']
        )
        return self.consolidate_subclasses(df) if consolidate_subclasses else df

    # Informe columns that add up across the subclasses of one class. Checked on
    # the June 2026 informe against the class-level PL in CDA's PL file: for the
    # classes with several subclasses, the sum of the subclasses' VL_TOTAL or of
    # their VL_PATRIM_LIQ is within 0.01% of it for 89% of them (the largest
    # subclass alone, for 4%). VL_QUOTA is absent on purpose: it is a price.
    _SUBCLASS_ADDITIVE_COLUMNS = ('VL_TOTAL', 'VL_PATRIM_LIQ', 'CAPTC_DIA', 'RESG_DIA', 'NR_COTST')

    def consolidate_subclasses(self, df: pd.DataFrame) -> pd.DataFrame:
        """Give each class ONE informe row per day instead of one per subclass.

        A class with subclasses reports a row per subclass per day under the
        class CNPJ, each with its own quota (Kinea Absoluto: 2.05, 2.05, 2.05,
        1.01), PL and holder count. Left as is, every analyzer that groups by
        ``CNPJ_FUNDO`` reads those series interleaved: quota "returns" are the
        gaps between subclasses (31.6% exceed 5% in the June 2026 informe on the
        221 classes concerned), PL and holders saw-tooth, and every Benford
        sample repeats once per subclass.

        On a class-day with several rows this returns one row where

        - VL_TOTAL, VL_PATRIM_LIQ, CAPTC_DIA, RESG_DIA and NR_COTST are the SUM
          over the subclasses (DERIVED). NR_COTST is then an upper bound on
          distinct holders (a holder of two subclasses counts twice). The flows
          include transfers between subclasses, which net out in FLUXO_LIQ_DIA.
        - VL_QUOTA is the quota the DOMINANT subclass reported that day (REPORTED,
          not derived). Dominant = the class's subclass with the largest PL
          summed over the rows given, so one series is followed for the whole
          window. It is NaN on a day that subclass did not report; another
          subclass's quota would be a price jump that never happened.
        - N_SUBCLASS_ROWS counts the rows folded; ID_SUBCLASSE_QUOTA names the
          subclass the quota comes from; ID_SUBCLASSE is blank.

        Every other row passes through untouched, in its place (N_SUBCLASS_ROWS
        = 1). CVM publishes no class-level row beside subclass rows (0 cases in
        June 2026); should one appear on a day, it alone is kept for that day, so
        nothing is counted twice.
        """
        key = ['CNPJ_FUNDO', 'DT_COMPTC']
        if df.empty or not {*key, 'ID_SUBCLASSE', 'VL_QUOTA', 'VL_PATRIM_LIQ'} <= set(df.columns):
            return df

        df = df.reset_index(drop=True)
        df['ID_SUBCLASSE'] = df['ID_SUBCLASSE'].astype('string').str.strip().replace('', pd.NA)
        class_level = df['ID_SUBCLASSE'].isna()
        on_class_day = class_level.groupby([df[k] for k in key], dropna=False).transform('any')
        if (on_class_day & ~class_level).any():
            logger.warning(
                "Informe: dropped %d subclass row(s) on days when the class also reports "
                "a class-level row", int((on_class_day & ~class_level).sum()))
            df = df[~(on_class_day & ~class_level)]

        # N_SUBCLASS_ROWS and ID_SUBCLASSE_QUOTA are the provenance of the folded rows.
        df = df.assign(N_SUBCLASS_ROWS=1, ID_SUBCLASSE_QUOTA=df['ID_SUBCLASSE'])
        dated = df[key].notna().all(axis=1)
        # Every row of a class that ever has several rows on one day is folded,
        # also its single-row days, so that one quota series runs through the window.
        several = df.duplicated(key, keep=False) & dated & df['ID_SUBCLASSE'].notna()
        scope = df['CNPJ_FUNDO'].isin(df.loc[several, 'CNPJ_FUNDO']) & dated
        if not scope.any():
            return df

        rows = df[scope]
        additive = [c for c in self._SUBCLASS_ADDITIVE_COLUMNS if c in rows.columns]
        by_day = rows.groupby(key, sort=False)
        folded = by_day[additive].sum(min_count=1)
        folded['N_SUBCLASS_ROWS'] = by_day.size()
        folded['_row'] = rows.assign(_row=rows.index).groupby(key, sort=False)['_row'].min()
        for col in rows.columns.difference([*key, *additive, 'ID_SUBCLASSE', 'ID_SUBCLASSE_QUOTA',
                                            'N_SUBCLASS_ROWS', 'VL_QUOTA']):
            folded[col] = by_day[col].first()

        # Follow one subclass per class: the one holding the most PL in these rows.
        pl = rows.groupby(['CNPJ_FUNDO', 'ID_SUBCLASSE'], as_index=False)['VL_PATRIM_LIQ'].sum()
        dominant = (pl.sort_values(['VL_PATRIM_LIQ', 'ID_SUBCLASSE'])
                    .drop_duplicates('CNPJ_FUNDO', keep='last')
                    .rename(columns={'ID_SUBCLASSE': 'DOMINANT'})[['CNPJ_FUNDO', 'DOMINANT']])
        rows = rows.merge(dominant, on='CNPJ_FUNDO', how='left')
        # A class-level row (no subclass id) reports the class's own quota.
        source = rows[(rows['ID_SUBCLASSE'] == rows['DOMINANT']) | rows['ID_SUBCLASSE'].isna()]
        source = source.drop_duplicates(key).set_index(key)[['VL_QUOTA', 'ID_SUBCLASSE_QUOTA']]
        folded = folded.join(source)
        folded = folded.reset_index().set_index('_row')

        out = pd.concat([df[~scope], folded]).sort_index(kind='stable').reset_index(drop=True)
        return out[list(df.columns)]

    # Cada bloco do CDA identifica o ativo por uma coluna diferente, conforme a
    # classe: BLC_4 e fie trazem CD_ATIVO, BLC_1 titulos publicos (CD_SELIC),
    # BLC_2 cotas de fundos (CNPJ do fundo investido), BLC_3 swaps, BLC_7
    # ativos no exterior. Os blocos de credito privado (5, 6, 8) so identificam
    # pelo emissor mais a descricao do papel. A ordem aqui e de preferencia:
    # o primeiro campo presente e nao vazio vira CD_ATIVO.
    _CDA_ASSET_ID_COLUMNS = (
        'CD_ATIVO',
        'CD_ATIVO_BV_MERC',
        'CD_ISIN',
        'CD_SELIC',
        'CD_SWAP',
        'CNPJ_FUNDO_CLASSE_COTA',
        'DS_ATIVO',
        'DS_SWAP',
        # Ultimo recurso: BLC_5 e BLC_6 (depositos a prazo e outros creditos
        # privados, ~41 mil posicoes/mes) nao trazem codigo de ativo nenhum. Para
        # credito privado o emissor e a identidade relevante de qualquer forma --
        # e o que se quer agrupar ao procurar exposicao concentrada ou marcacao
        # divergente. Sem isto essas linhas ficariam sem identificador.
        'EMISSOR',
    )

    def read_cda(self, file_path: Path,
                 encoding: str = 'latin1',
                 sep: str = ';') -> pd.DataFrame:
        """Lê um bloco do CDA (Composição de Carteira) e normaliza suas colunas.

        Aceita qualquer um dos CSVs que a CVM publica dentro do ZIP mensal. Os
        nomes das colunas de posicao variam por bloco, entao aqui eles sao
        levados ao formato que os analisadores esperam: VL_MERCADO, QT_POS e
        CD_ATIVO.
        """
        df = self._read_csv_generic(
            file_path=file_path,
            file_type="CDA",
            encoding=encoding,
            sep=sep,
            date_cols=['DT_COMPTC'],
            cnpj_cols=['CNPJ_FUNDO', 'CNPJ_EMISSOR', 'CNPJ_FUNDO_CLASSE_COTA'],
            numeric_cols=['VL_MERC_POS_FINAL', 'QT_POS_FINAL', 'VL_CUSTO_POS_FINAL'],
        )

        return self._normalize_cda_positions(df)

    @classmethod
    def _normalize_cda_positions(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Dá aos blocos do CDA um formato comum: VL_MERCADO, QT_POS, CD_ATIVO."""
        if df.empty:
            return df

        df = df.copy()

        # Nomes canonicos de posicao usados por todos os analisadores.
        if 'VL_MERCADO' not in df.columns and 'VL_MERC_POS_FINAL' in df.columns:
            df['VL_MERCADO'] = df['VL_MERC_POS_FINAL']
        if 'QT_POS' not in df.columns and 'QT_POS_FINAL' in df.columns:
            df['QT_POS'] = df['QT_POS_FINAL']

        # A funcao e idempotente de proposito: os blocos passam por aqui ao
        # serem lidos e o frame concatenado passa de novo. Sem esta guarda a
        # segunda passagem veria CD_ATIVO ja preenchido e marcaria toda a
        # procedencia como 'CD_ATIVO', apagando o registro de quais linhas
        # foram identificadas so pelo emissor.
        if 'CD_ATIVO_FONTE' in df.columns:
            return df

        # Identificador do ativo: primeiro campo disponivel e nao vazio.
        if 'CD_ATIVO' not in df.columns:
            df['CD_ATIVO'] = pd.NA
        asset_id = df['CD_ATIVO'].astype('string')
        source = pd.Series(pd.NA, index=df.index, dtype='string')
        source = source.where(asset_id.isna(), 'CD_ATIVO')

        for column in cls._CDA_ASSET_ID_COLUMNS:
            if column == 'CD_ATIVO' or column not in df.columns:
                continue
            candidate = df[column].astype('string').str.strip()
            candidate = candidate.where(candidate.str.len() > 0)
            newly_filled = asset_id.isna() & candidate.notna()
            asset_id = asset_id.fillna(candidate)
            source = source.mask(newly_filled, column)

        df['CD_ATIVO'] = asset_id
        # Qual coluna deu a identidade. Importa porque nem toda identidade e
        # equivalente: EMISSOR identifica quem emitiu, nao qual papel. Um banco
        # emite dezenas de CDBs com vencimentos e taxas diferentes, entao
        # comparar precos entre linhas identificadas pelo emissor compara papeis
        # distintos. Analises de preco devem usar so identidade de instrumento.
        df['CD_ATIVO_FONTE'] = source

        return df

    #: Colunas de identidade que designam um instrumento especifico, nao apenas
    #: seu emissor. Só estas permitem comparar preços entre fundos.
    INSTRUMENT_LEVEL_ASSET_SOURCES = frozenset({
        'CD_ATIVO', 'CD_ATIVO_BV_MERC', 'CD_ISIN', 'CD_SELIC', 'CD_SWAP',
        'CNPJ_FUNDO_CLASSE_COTA', 'DS_ATIVO', 'DS_SWAP',
    })

    def read_cadastro(self, file_path: Path,
                     encoding: str = 'latin1',
                     sep: str = ';') -> pd.DataFrame:
        """Lê arquivo de Cadastro de Fundos"""
        return self._read_csv_generic(
            file_path=file_path,
            file_type="Cadastro",
            encoding=encoding,
            sep=sep,
            date_cols=['DT_REG', 'DT_CONST', 'DT_CANCEL', 'DT_INI_SIT', 'DT_INI_ATIV', 'DT_INI_EXERC', 'DT_FIM_EXERC'],
            cnpj_cols=['CNPJ_FUNDO', 'CNPJ_ADMIN', 'CNPJ_GESTOR'],
            numeric_cols=None
        )

    #: registro_classe/registro_fundo -> nomes canonicos usados pelos analisadores.
    _REGISTRO_COLUMN_MAP = {
        'CNPJ_CLASSE': 'CNPJ_FUNDO',          # o informe reporta pela classe
        'CNPJ_FUNDO': 'CNPJ_FUNDO_PAI',       # o fundo que abriga a classe
        'DENOMINACAO_SOCIAL': 'DENOM_SOCIAL',
        'CNPJ_ADMINISTRADOR': 'CNPJ_ADMIN',
        'ADMINISTRADOR': 'ADMIN',
        'CPF_CNPJ_GESTOR': 'CNPJ_GESTOR',
        'GESTOR': 'GESTOR',
        'SITUACAO': 'SIT',
        'CLASSIFICACAO': 'CLASSE',
        'DATA_CONSTITUICAO': 'DT_CONST',
        'DATA_REGISTRO': 'DT_REG',
    }

    def read_registro_fundo_classe(self, path: Path) -> pd.DataFrame:
        """Lê o registro RCVM 175 e devolve o formato canonico de cadastro.

        Junta registro_classe (a entidade que o informe diario reporta) com
        registro_fundo (que carrega administrador e gestor) e renomeia as
        colunas para o vocabulario que os analisadores ja usam, de modo que
        nada a jusante precise saber qual dos dois cadastros foi carregado.

        Args:
            path: Diretorio contendo registro_classe.csv e registro_fundo.csv,
                ou o caminho de registro_classe.csv

        Returns:
            DataFrame de cadastro, ou vazio se os arquivos nao existirem
        """
        directory = path if path.is_dir() else path.parent
        classe_path = directory / 'registro_classe.csv'
        fundo_path = directory / 'registro_fundo.csv'

        if not classe_path.exists():
            logger.warning(f"registro_classe.csv nao encontrado em {directory}")
            return pd.DataFrame()

        classe = self._read_csv_generic(classe_path, file_type="Registro Classe")
        if classe.empty:
            return classe

        if fundo_path.exists():
            fundo = self._read_csv_generic(fundo_path, file_type="Registro Fundo")
            if not fundo.empty and 'ID_REGISTRO_FUNDO' in fundo.columns:
                # Só o registro do fundo traz administrador e gestor.
                keep = [
                    c for c in (
                        'ID_REGISTRO_FUNDO', 'CNPJ_FUNDO', 'CNPJ_ADMINISTRADOR',
                        'ADMINISTRADOR', 'CPF_CNPJ_GESTOR', 'GESTOR',
                    ) if c in fundo.columns
                ]
                classe = classe.merge(
                    fundo[keep], on='ID_REGISTRO_FUNDO', how='left', suffixes=('', '_FUNDO')
                )
        else:
            logger.warning("registro_fundo.csv ausente: sem administrador/gestor")

        df = classe.rename(columns={
            old: new for old, new in self._REGISTRO_COLUMN_MAP.items()
            if old in classe.columns
        })

        for col in ('CNPJ_FUNDO', 'CNPJ_FUNDO_PAI', 'CNPJ_ADMIN', 'CNPJ_GESTOR'):
            if col in df.columns:
                df[col] = normalize_cnpj_series(df[col])

        for col in ('DT_CONST', 'DT_REG'):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # O registro novo escreve "Em Funcionamento Normal"; o legado usava
        # caixa alta. Uniformiza para que os filtros de situacao valham para os
        # dois formatos.
        if 'SIT' in df.columns:
            df['SIT'] = df['SIT'].astype('string').str.upper()

        logger.info(f"Registro carregado: {len(df):,} classes")
        return df

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares DataFrame by normalizing columns and applying aliases.
        This is a helper to avoid duplicate normalization in filter methods.
        """
        df = self._normalize_columns(df)
        df = self._apply_column_aliases(df)
        return df

    def filter_by_cnpj(self, df: pd.DataFrame, cnpj_list: list[str]) -> pd.DataFrame:
        """Filtra DataFrame por lista de CNPJs"""
        df = self._prepare_dataframe(df)
        if 'CNPJ_FUNDO' not in df.columns:
            return pd.DataFrame()

        normalized_list = self._normalize_cnpj_list(cnpj_list)
        normalized_series = self._normalize_cnpj_series(df['CNPJ_FUNDO'])
        return df[normalized_series.isin(normalized_list)].copy()

    def filter_by_administrador(self, df: pd.DataFrame, admin_cnpj_list: list[str]) -> pd.DataFrame:
        """Filtra DataFrame por CNPJ do administrador"""
        df = self._prepare_dataframe(df)
        if 'CNPJ_ADMIN' not in df.columns:
            return pd.DataFrame()

        normalized_list = self._normalize_cnpj_list(admin_cnpj_list)
        normalized_series = self._normalize_cnpj_series(df['CNPJ_ADMIN'])
        return df[normalized_series.isin(normalized_list)].copy()

    def filter_by_gestor(self, df: pd.DataFrame, gestor_cnpj_list: list[str]) -> pd.DataFrame:
        """Filtra DataFrame por CNPJ do gestor"""
        df = self._prepare_dataframe(df)
        if 'CNPJ_GESTOR' not in df.columns:
            return pd.DataFrame()

        normalized_list = self._normalize_cnpj_list(gestor_cnpj_list)
        normalized_series = self._normalize_cnpj_series(df['CNPJ_GESTOR'])
        return df[normalized_series.isin(normalized_list)].copy()

    def filter_by_date_range(self, df: pd.DataFrame,
                            start_date: str,
                            end_date: str,
                            date_col: str = 'DT_COMPTC') -> pd.DataFrame:
        """Filtra DataFrame por intervalo de datas"""
        df = self._prepare_dataframe(df)
        if date_col not in df.columns:
            return pd.DataFrame()

        mask = (df[date_col] >= start_date) & (df[date_col] <= end_date)
        return df[mask].copy()

    def calculate_net_flow(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula fluxo líquido (captação - resgate)"""
        if 'CAPTC_DIA' in df.columns and 'RESG_DIA' in df.columns:
            df = df.copy()
            df['FLUXO_LIQ_DIA'] = df['CAPTC_DIA'] - df['RESG_DIA']
        return df

    def aggregate_by_fund(self, df: pd.DataFrame,
                         agg_dict: dict | None = None) -> pd.DataFrame:
        """Agrega dados por fundo"""
        if agg_dict is None:
            agg_dict = {
                'VL_TOTAL': 'sum',
                'VL_PATRIM_LIQ': 'last',
                'CAPTC_DIA': 'sum',
                'RESG_DIA': 'sum',
                'FLUXO_LIQ_DIA': 'sum',
                'NR_COTST': 'last'
            }

        # Filtrar apenas colunas que existem
        valid_agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}

        if 'CNPJ_FUNDO' in df.columns:
            return df.groupby('CNPJ_FUNDO').agg(valid_agg_dict).reset_index()

        return df

    def save_processed(self, df: pd.DataFrame, filename: str):
        """Salva dados processados"""
        output_path = self.config.PROCESSED_DATA_DIR / filename
        df.to_csv(output_path, index=False, encoding='utf-8', sep=';')
        logger.info(f"Dados salvos em: {output_path}")
        return output_path

    def filter_by_fund_list(self, df: pd.DataFrame, cnpj_list: list[str]) -> pd.DataFrame:
        """
        Filter DataFrame by fund CNPJ list.

        Alias for filter_by_cnpj for consistency with new fund selector terminology.

        Args:
            df: DataFrame to filter
            cnpj_list: List of fund CNPJs

        Returns:
            Filtered DataFrame
        """
        return self.filter_by_cnpj(df, cnpj_list)

    def get_fund_metadata(self, cadastro_df: pd.DataFrame, cnpj_list: list[str]) -> pd.DataFrame:
        """
        Extract fund metadata for specified CNPJs.

        Args:
            cadastro_df: Cadastro DataFrame
            cnpj_list: List of fund CNPJs

        Returns:
            DataFrame with fund metadata (name, status, administrator, etc.)
        """
        filtered = self.filter_by_cnpj(cadastro_df, cnpj_list)

        # Select relevant columns if they exist
        metadata_cols = ['CNPJ_FUNDO', 'DENOM_SOCIAL', 'SIT', 'CNPJ_ADMIN', 'ADMIN',
                        'CNPJ_GESTOR', 'GESTOR', 'TP_FUNDO', 'CLASSE']
        available_cols = [col for col in metadata_cols if col in filtered.columns]

        if available_cols:
            return filtered[available_cols].copy()
        return filtered

    def merge_fund_metadata(self, df: pd.DataFrame, cadastro_df: pd.DataFrame,
                           cols_to_merge: list[str] | None = None) -> pd.DataFrame:
        """
        Enrich DataFrame with fund metadata from cadastro.

        Args:
            df: DataFrame to enrich (must have CNPJ_FUNDO column)
            cadastro_df: Cadastro DataFrame with fund metadata
            cols_to_merge: Specific columns to merge. If None, merges common useful columns.

        Returns:
            Enriched DataFrame with metadata
        """
        if 'CNPJ_FUNDO' not in df.columns or 'CNPJ_FUNDO' not in cadastro_df.columns:
            logger.warning("CNPJ_FUNDO column missing, cannot merge metadata")
            return df.copy()

        # Default columns to merge
        if cols_to_merge is None:
            cols_to_merge = ['DENOM_SOCIAL', 'SIT', 'ADMIN', 'GESTOR']

        # Filter to only available columns
        available_cols = ['CNPJ_FUNDO'] + [col for col in cols_to_merge if col in cadastro_df.columns]

        # Prepare cadastro subset
        cadastro_subset = cadastro_df[available_cols].drop_duplicates('CNPJ_FUNDO')

        # Merge
        result = df.merge(cadastro_subset, on='CNPJ_FUNDO', how='left', suffixes=('', '_metadata'))

        return result
