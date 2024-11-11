import sys
import io
import json
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def load_and_clean_data(file_path):
    """
    加载并清洗电影数据集
    :param file_path: 字符串，表示电影数据集的路径
    :return: 返回一个 Pandas DataFrame，包含清洗后的电影数据集
    """

    # 读取 CSV 文件中的数据
    movie_df = pd.read_csv(file_path)  # 读取原始数据集
    # 删除评分为0的电影，筛选出实际有评分的电影
    movie_df = movie_df[movie_df['vote_average'] != 0]
    # 重置数据框的索引，使得索引从0开始
    movie_df.reset_index(inplace=True)
    # print(movie_df.shape)  # 打印数据集的维度（行数和列数）

    # 删除不需要的列，这些列对后续分析和模型训练没有用处
    movie_df_cleaned = movie_df.drop(['index', 'vote_count', 'status', 'release_date', 'revenue', 'backdrop_path',
                                      'budget', 'homepage', 'imdb_id', 'original_title', 'overview', 'poster_path',
                                      'tagline', 'production_companies', 'production_countries', 'spoken_languages',
                                      'keywords'], axis=1)

    # 保存原始的电影标题，以便后续使用
    movie_df_cleaned['original_title'] = movie_df_cleaned['title']

    # 填充缺失的genres（电影类型）列，缺失值填充为'unknown'
    movie_df_cleaned['genres'] = movie_df_cleaned['genres'].fillna('unknown')

    # 检查清洗后的数据中是否存在缺失值
    # print(movie_df_cleaned.isna().sum())

    # 去除重复的电影数据
    movie_df_cleaned = movie_df_cleaned.drop_duplicates()

    # 创建一个副本用于后续处理
    movie_df_final = movie_df_cleaned.copy()

    return movie_df_final


def encode_genres(movie_df):
    """
    对电影类型进行编码，转化为二进制特征
    :param movie_df: 一个 Pandas DataFrame，包含了清洗后的电影数据
    :return: 返回一个 Pandas DataFrame，包含了编码后的电影数据
    """

    # 将电影类型（genres）列中的字符串分割成列表
    genres_list = movie_df['genres'].apply(lambda x: x.split(','))
    # 将分割后的电影类型转换为 DataFrame
    genres_list = pd.DataFrame(genres_list)

    # 清洗电影类型数据：去除每个类型的空格、转为小写并去掉空格
    genres_list['genres'] = genres_list['genres'].apply(lambda x: [y.strip().lower().replace(' ', '') for y in x])

    # 使用 MultiLabelBinarizer 对电影类型进行二进制编码
    mlb = MultiLabelBinarizer()
    genre_encoded = mlb.fit_transform(genres_list['genres'])

    # 将编码后的结果转换为 DataFrame
    genre_encoded_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)
    # 重置索引
    genre_encoded_df = genre_encoded_df.reset_index()

    # 删除原始的 genres 列
    movie_df_final_cleaned = movie_df.drop(['genres'], axis=1)
    # 重置索引
    movie_df_final_cleaned = movie_df_final_cleaned.reset_index()

    # 合并编码后的电影类型数据与原始数据
    movie_df_final = pd.concat([movie_df_final_cleaned, genre_encoded_df], axis=1).drop('index', axis=1)

    return movie_df_final


def clean_titles_and_languages(movie_df):
    """
    清洗电影标题和语言信息
    :param movie_df: 一个 Pandas DataFrame，包含了电影数据
    :return: 返回一个 Pandas DataFrame，包含了清洗后的电影数据
    """

    # 清洗电影标题：去除空格、转为小写字母并去除所有空格
    movie_df['title'] = movie_df['title'].apply(lambda x: x.strip().lower().replace(' ', ''))
    # 清洗电影原始语言：去除空格、转为小写字母并去除所有空格
    movie_df['original_language'] = movie_df['original_language'].apply(lambda x: x.strip().lower().replace(' ', ''))

    # 将原始语言列中的非主要语言（如'cn', 'ja', 'kr', 'en'）归类为“else”
    movie_df.loc[~((movie_df['original_language'] == 'en') | (movie_df['original_language'] == 'cn') |
                   (movie_df['original_language'] == 'ja') | (
                           movie_df['original_language'] == 'kr')), 'original_language'] = 'else'

    return movie_df


def encode_adult_and_language(movie_df):
    """
    对成人内容和语言信息进行编码
    :param movie_df: 一个 Pandas DataFrame，包含了电影数据
    :return: 返回一个 Pandas DataFrame，包含了编码后的电影数据
    """

    # 对成人标签进行编码：'True' 或 'False'，转为字符串格式
    ohe = OneHotEncoder(sparse_output=False)
    movie_df['adult'] = movie_df['adult'].astype('str')
    # 使用 OneHotEncoder 对成人内容进行编码
    adult_encoded = ohe.fit_transform(movie_df[['adult']])
    adult_encoded_df = pd.DataFrame(adult_encoded, columns=ohe.get_feature_names_out())
    # 删除 'adult_True' 这一列（我们关心的是非成人内容）
    adult_encoded_df = adult_encoded_df.drop('adult_True', axis=1)

    # 对原始语言进行 OneHot 编码
    language_encoded = ohe.fit_transform(movie_df[['original_language']])
    language_encoded_df = pd.DataFrame(language_encoded, columns=ohe.get_feature_names_out())

    # 删除原始的成人内容和原始语言列
    movie_df_final_cleaned = movie_df.drop(['adult', 'original_language'], axis=1)

    # 合并编码后的成人内容和语言数据
    movie_df_final = pd.concat([movie_df_final_cleaned, adult_encoded_df, language_encoded_df], axis=1)

    return movie_df_final


def normalize_data(movie_df):
    """
    对数值数据进行标准化
    :param movie_df: 一个 Pandas DataFrame，包含了电影数据
    :return: 返回一个 Pandas DataFrame，包含了标准化后的电影数据
    """

    # 导入 StandardScaler 用于数据标准化
    from sklearn.preprocessing import StandardScaler

    # 提取 id 列
    movie_id = movie_df['id']

    # 初始化标准化器
    scaler = StandardScaler()
    # 对除电影标题和 id 外的所有数值型列进行标准化处理
    movie_df_norm = scaler.fit_transform(movie_df.drop(['title', 'original_title', 'id'], axis=1))

    # 将标准化后的结果转换为 DataFrame
    movie_df_norm_df = pd.DataFrame(movie_df_norm,
                                    columns=[x for x in movie_df.columns if x not in ['title', 'original_title', 'id']])

    # 合并标准化后的数值数据、原始电影标题和 id 列
    movie_df_final = pd.concat([movie_id, movie_df[['title', 'original_title']], movie_df_norm_df], axis=1)

    return movie_df_final


def get_movie_recommendations(movie_name, movie_df):
    """
    根据电影名称获取推荐的电影列表
    :param movie_name: 一个字符串，表示电影的标题
    :param movie_df: 一个 Pandas DataFrame，包含了电影数据
    :return: 返回一个 Pandas DataFrame，包含了前20部推荐的电影
    """

    # 删除重复的电影标题
    movie_df_final = movie_df.drop_duplicates(subset=['title'])

    # 设置电影标题列为索引，以便后续查找
    movie_df_final = movie_df_final.set_index(['title'])

    # 删除原始标题列
    movie_df_final_final = movie_df_final.drop('original_title', axis=1)

    # 清理目标电影的标题
    movie_name = movie_name.strip().lower().replace(' ', '')
    # 获取目标电影的数据
    movie_data = movie_df_final_final.loc[[movie_name]]

    # 将目标电影的数据转换为二维数组
    movie_data = movie_data.values.reshape(1, -1)

    # 排除目标电影的数据，计算其他电影与目标电影的相似度
    movie_df_other = movie_df_final_final.loc[movie_df_final.index != movie_name, :]
    # 获取除目标电影外的所有电影标题以及电影id
    movie_titles = movie_df_final.loc[movie_df_final.index != movie_name, ['id', 'original_title']]

    # 计算余弦相似度
    cosine_sim_matrix = cosine_similarity(movie_data, movie_df_other)
    # 将余弦相似度矩阵转换为 DataFrame
    cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=[movie_name], columns=movie_titles)

    # 打印相似度矩阵（可选）
    # print(cosine_sim_df)

    # 获取与目标电影最相似的前20部电影
    sorted_similar_movies = cosine_sim_df.loc[movie_name].sort_values(ascending=False)[0:20]

    # 打印最相似的电影列表
    # print(sorted_similar_movies)

    return sorted_similar_movies

def get_recommendation_by_id(movie_id, movie_df):
    """
    根据电影ID获取推荐的电影列表
    :param movie_id: 一个整数，表示电影的ID
    :param movie_df: 一个 Pandas DataFrame，包含了电影数据
    :return: 返回一个 Pandas DataFrame，包含了前20部推荐的电影
    """

    # 删除重复的电影ID
    movie_df_final = movie_df.drop_duplicates(subset=['id'])

    # 设置电影ID列为索引，以便后续查找
    movie_df_final = movie_df_final.set_index(['id'])

    # 获取目标电影的数据
    movie_data = movie_df_final.loc[[movie_id]]

    # 将目标电影的数据转换为二维数组
    movie_data = movie_data.values.reshape(1, -1)

    # 排除目标电影的数据，计算其他电影与目标电影的相似度
    movie_df_other = movie_df_final.loc[movie_df_final.index != movie_id, :]
    # 获取除目标电影外的所有电影标题以及电影id
    movie_titles = movie_df_final.loc[movie_df_final.index != movie_id, ['original_title']]

    # 只保留数值型列用于相似度计算
    movie_data_numeric = movie_data[:, 2:]
    movie_df_other_numeric = movie_df_other.iloc[:, 2:]

    # 计算余弦相似度
    cosine_sim_matrix = cosine_similarity(movie_data_numeric, movie_df_other_numeric)

    # 将余弦相似度矩阵转换为 DataFrame
    cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=[movie_id], columns=movie_titles.index)

    # 获取与目标电影最相似的前20部电影
    sorted_similar_movies = cosine_sim_df.loc[movie_id].sort_values(ascending=False)[0:20]

    return sorted_similar_movies

if __name__ == '__main__':
    import sys

    # 从命令行参数获取电影名称
    movie_name = 'the dark knight'

    # 加载并清理数据
    movie_data = load_and_clean_data('../dataset/TMDB_movie_dataset_v11.csv')

    # 对电影类型进行编码
    movie_data_encoded = encode_genres(movie_data)

    # 清理电影标题和语言信息
    movie_data_cleaned = clean_titles_and_languages(movie_data_encoded)

    # 对成人内容和语言进行编码
    movie_data_final = encode_adult_and_language(movie_data_cleaned)

    # 标准化数据
    movie_data_normalized = normalize_data(movie_data_final)

    # 获取推荐结果
    recommended_movies = get_movie_recommendations(movie_name, movie_data_normalized)
    recommended_movies_by_id = get_recommendation_by_id(238, movie_data_normalized)

    # Convert the keys to strings before dumping to JSON
    recommended_movies_dict = {str(k): v for k, v in recommended_movies.to_dict().items()}
    recommended_movies_by_id_dict = {str(k): v for k, v in recommended_movies_by_id.to_dict().items()}

    # print(json.dumps(recommended_movies_dict))
    print(json.dumps(recommended_movies_by_id_dict))