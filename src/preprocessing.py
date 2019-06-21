
import pandas as pd
from pathlib import Path
import nlp_preprocessing
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger.setLevel(logging.INFO)

dir = Path(__file__).parent.parent


def load_file(train):
    train_df = pd.read_csv(train, delimiter='\t', encoding='utf-8')
    return train_df


def __prepare(df):
    # exclude lead paragraph section from aspect candidates
    df.drop(df[(df["aspect"] == 'lead_paragraphs')].index, inplace=True, axis=0)
    df.dropna(inplace=True)
    # preprocessing the text

    cols = ["sentence", "aspect_content", "aspect"]
    for index, row in df.iterrows():
        for col in cols:
            text = nlp_preprocessing.extra_cleaning_text(row[col])
            df.at[index, col] = text

    for col in cols:
        df[col] = df[col].map(nlp_preprocessing.nlp_pipeline, na_action='ignore')

    df.drop(df[(df.aspect_content == '')].index, inplace=True, axis=0)
    df.drop(df[(df.aspect == '')].index, inplace=True, axis=0)
    # df[cols] = df[cols].apply(nlp_preprocessing.whitespace_tokenize)
    df.index = range(len(df.index))

    return df


# In[39]:
if __name__ == '__main__':

    test_file = Path.joinpath(dir, 'trained', 'labeled_both_subj.tsv')
    f1 = Path.joinpath(dir, 'trained', 'labeled_subj.tsv')
    f2 = Path.joinpath(dir, 'trained', 'labeled_obj.tsv')
    f3 = Path.joinpath(dir, 'trained', 'labeled_both_obj.tsv')
    test_df = pd.read_csv(test_file, delimiter='\t', encoding='utf-8')
    df1 = pd.read_csv(f1, delimiter='\t', encoding='utf-8')
    df2 = pd.read_csv(f2, delimiter='\t', encoding='utf-8')
    df3 = pd.read_csv(f3, delimiter='\t', encoding='utf-8')
    logging.info("File loaded.")
    test_df_cleaned = __prepare(test_df)
    test_df_cleaned.to_csv(Path.joinpath(dir, "trained", "cleaned_both_subj.tsv"), index=False, sep="\t",
                          encoding='utf-8')
    df1_cleaned = __prepare(df1)

    df1_cleaned.to_csv(Path.joinpath(dir, "trained", "cleaned_subj.tsv"), index=False,  sep="\t", encoding='utf-8')
    logging.info('preprocessed text successfully.')
    df2_cleaned = __prepare(df2)
    df2_cleaned.to_csv(Path.joinpath(dir, "trained", "cleaned_obj.tsv"), index=False, sep="\t",
                           encoding='utf-8')
    logging.info('preprocessed text successfully.')
    df3_cleaned = __prepare(df3)
    df3_cleaned.to_csv(Path.joinpath(dir, "trained", "cleaned_both_obj.tsv"), index=False, sep="\t",
                           encoding='utf-8')
    logging.info('preprocessed text successfully.')

    logging.info("cleaned file saved")