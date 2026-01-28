import re
import datetime
import requests
from bs4 import BeautifulSoup
from collections import Counter, defaultdict

class Movies:
    """
    analyzing movies.csv
    """
    def __init__(self, path_to_the_file):
        self.file = path_to_the_file
        self.rows_count = 1000

    def dist_by_release_year(self):
        lines_generator = Aux.csv_to_generator(self.file, self.rows_count)
        release_years = dict()
        year_pattern = re.compile(r'\(\d{4}\)$')

        for line in lines_generator:
            _, title, _ = Aux.parse_csv_line(line)
            year = re.search(year_pattern, title).group(0).strip('()')
            release_years[year] = release_years.get(year, 0) + 1

        release_years = dict(sorted(release_years.items(), key=lambda x: x[1], reverse=True))

        return release_years

    def dist_by_genres(self):
        lines_generator = Aux.csv_to_generator(self.file, self.rows_count)
        genres = dict()

        for line in lines_generator:
            _, _, genre_list = Aux.parse_csv_line(line)
            for genre in genre_list.split('|'):
                genre = genre.strip()
                genres[genre] = genres.get(genre, 0) + 1

        genres = dict(sorted(genres.items(), key=lambda x: x[1], reverse=True))

        return genres

    def most_genres(self, n=None):
        """
        top movies by number of genres
        """
        n = (self.rows_count if n is None else n)
        lines_generator = Aux.csv_to_generator(self.file, self.rows_count)
        movies = dict()

        for line in lines_generator:
            _, title, genre_list = Aux.parse_csv_line(line)
            movies[title] = len(genre_list.split('|'))

        movies = dict(sorted(movies.items(), key=lambda x: x[1], reverse=True)[:n])

        return movies


class Tags:
    """
    analyzing tags.csv
    """
    def __init__(self, path_to_the_file):
        self.file = path_to_the_file
        self.rows_count = 1000

    def most_words(self, n=None):
        """
        top tags by word count
        """
        n = (self.rows_count if n is None else n)
        lines_generator = Aux.csv_to_generator(self.file, self.rows_count)
        big_tags = dict()
        word_pattern = re.compile(r"\b([\w'-]+(?:\(s\))?)")

        for line in lines_generator:
            _, _, tag, _ = Aux.parse_csv_line(line)
            big_tags[tag] = len(re.findall(word_pattern, tag))

        big_tags = dict(sorted(big_tags.items(), key=lambda x: x[1], reverse=True)[:n])

        return big_tags

    def longest(self, n=None):
        """
        top tags by character length
        """
        n = (self.rows_count if n is None else n)
        lines_generator = Aux.csv_to_generator(self.file, self.rows_count)
        big_tags = list(set([tag for _, _, tag, _ in (Aux.parse_csv_line(line) for line in lines_generator)]))

        big_tags.sort(key=lambda x: len(x), reverse=True)

        return big_tags[:n]

    def most_words_and_longest(self, n=None):
        """
        intersection between top-n tags with most words and top-n longest tags by characters
        """
        n = (self.rows_count if n is None else n)
        most_words = self.most_words(n)
        longest = self.longest(n)

        big_tags = list(set(most_words) & set(longest))
        big_tags.sort()

        return big_tags

    def most_popular(self, n=None):
        """
        most popular tags
        """
        n = (self.rows_count if n is None else n)
        lines_generator = Aux.csv_to_generator(self.file, self.rows_count)
        popular_tags = dict()

        for line in lines_generator:
            _, _, tag, _ = Aux.parse_csv_line(line)
            popular_tags[tag] = popular_tags.get(tag, 0) + 1

        popular_tags = dict(sorted(popular_tags.items(), key=lambda x: x[1], reverse=True)[:n])

        return popular_tags

    def tags_with(self, word):
        """
        all unique tags that include the word given
        """
        lines_generator = Aux.csv_to_generator(self.file, self.rows_count)
        tags_with_word = list(set([tag for _, _, tag, _ in (Aux.parse_csv_line(line) for line in lines_generator) if word.lower() in map(lambda x: x.lower(), tag.split())]))

        tags_with_word.sort()

        return tags_with_word

    def average_ratings_for_popular_tags(self, n_tags=10, ratings_object=None):
        """
        calculates avg rating of a tag. for analysis purposes, this function also does this for the most popular tags only
        """
        if ratings_object is None:
            raise ValueError("Ratings object is required for this method")

        popular_tags = self.most_popular(n_tags)

        tag_movies = defaultdict(set)
        lines_generator = Aux.csv_to_generator(self.file, self.rows_count)

        for line in lines_generator:
            _, movie_id, tag, _ = Aux.parse_csv_line(line)
            tag_movies[tag].add(movie_id)

        tag_avg_ratings = {}

        for tag in popular_tags:
            movie_ids = tag_movies.get(tag, set())
            ratings = []

            for movie_id in movie_ids:
                movie_ratings = ratings_object._get_movie_ratings(movie_id)
                ratings.extend(movie_ratings)

            if ratings:
                avg_rating = sum(ratings) / len(ratings)
                tag_avg_ratings[tag] = round(avg_rating, 2)
            else:
                tag_avg_ratings[tag] = 0.0

        return dict(sorted(tag_avg_ratings.items(), key=lambda x: x[1], reverse=True))


class Ratings:
    """
    analyzing ratings.csv
    """
    def __init__(self, path_to_ratings_file, path_to_movies_file):
        self.ratings_file = path_to_ratings_file
        self.movies_file = path_to_movies_file
        self.rows_count = 1000
        self.Movies_analyzer = self.Movies(self)
        self.Users_analyzer = self.Users(self)

    # -------------------------------------ratings distribution------------------------------------------------------#
    def ratings_dist_by_year(self):
        return self.Movies_analyzer.dist_by_year()

    def ratings_dist_by_value(self):
        return self.Movies_analyzer.dist_by_rating()

    # -------------------------------------ratings count by movie/user------------------------------------------------------#
    def top_movies_by_num_of_ratings(self, n=None):
        """
        top movies by the number of ratings sorted by numbers descendingly
        """
        n = (self.rows_count if n is None else n)
        return self.Movies_analyzer.top_by_num_of_ratings(n)

    def top_users_by_num_of_ratings(self, n=None):
        """
        top users by the number of ratings sorted by numbers descendingly
        """
        n = (self.rows_count if n is None else n)
        return self.Users_analyzer.top_by_num_of_ratings(n)

    # -------------------------------------movie avg/median rating----------------------------------------#
    def top_movies_by_average_rating(self, n=None):
        """
        top movies by average rating sorted by average rating descendingly
        """
        n = (self.rows_count if n is None else n)
        return self.Movies_analyzer.top_by_ratings(n, metric='average')

    def top_movies_by_median_rating(self, n=None):
        """
        top movies by median rating sorted by median rating descendingly
        """
        n = (self.rows_count if n is None else n)
        return self.Movies_analyzer.top_by_ratings(n, metric='median')

    # --------------------------------------user avg/median rating---------------------------------------------#
    def top_users_by_average_rating(self, n=None):
        """
        top users by average rating sorted by average rating descendingly
        """
        n = (self.rows_count if n is None else n)
        return self.Users_analyzer.top_by_ratings(n, metric='average')

    def top_users_by_median_rating(self, n=None):
        """
        top users by median rating sorted by median rating descendingly
        """
        n = (self.rows_count if n is None else n)
        return self.Users_analyzer.top_by_ratings(n, metric='median')

    # -----------------------------------------ratings variance-------------------------------------------------#
    def top_controversial_movies(self, n=None):
        """
        top movies by variance of the ratings sorted by variance descendingly
        """
        n = (self.rows_count if n is None else n)
        return self.Movies_analyzer.top_controversial(n)

    def top_controversial_users(self, n=None):
        """
        top users by variance of the ratings sorted by variance descendingly
        """
        n = (self.rows_count if n is None else n)
        return self.Users_analyzer.top_controversial(n)


    class Movies:

        def __init__(self, Ratings_instance):
            self.outer = Ratings_instance
            self.group_field = 1

        def dist_by_year(self):
            lines_generator = Aux.csv_to_generator(self.outer.ratings_file, self.outer.rows_count)
            years = []

            for line in lines_generator:
                _, _, _, timestamp = Aux.parse_csv_line(line)
                years.append(datetime.date.fromtimestamp(int(timestamp)).year)

            ratings_by_year = Counter(years)
            ratings_by_year = dict(sorted(ratings_by_year.items(), key=lambda x: int(x[0])))

            return ratings_by_year

        def dist_by_rating(self):
            lines_generator = Aux.csv_to_generator(self.outer.ratings_file, self.outer.rows_count)
            ratings = []

            for line in lines_generator:
                _, _, rating, _ = Aux.parse_csv_line(line)
                ratings.append(rating)

            ratings_distribution = Counter(ratings)
            ratings_distribution = dict(sorted(ratings_distribution.items(), key=lambda x: float(x[0])))

            return ratings_distribution

        def top_by_num_of_ratings(self, n):
            ratings_csv_generator = Aux.csv_to_generator(self.outer.ratings_file, self.outer.rows_count)

            top_by_field = dict()
            for line in ratings_csv_generator:
                field_value = Aux.parse_csv_line(line)[self.group_field]
                top_by_field[field_value] = top_by_field.get(field_value, 0) + 1

            if self.group_field == 1:
                top_by_field = {self.outer._get_title(k): v for k, v in top_by_field.items()}

            top_by_field = dict(sorted(top_by_field.items(), key=lambda x: x[1], reverse=True)[:n])

            return top_by_field

        def top_by_ratings(self, n, metric):
            ratings_csv_generator = Aux.csv_to_generator(self.outer.ratings_file, self.outer.rows_count)

            top_by_field = dict()
            for line in ratings_csv_generator:
                parsed_fields = Aux.parse_csv_line(line)
                group_field_value = parsed_fields[self.group_field]
                rating = float(parsed_fields[2])
                top_by_field[group_field_value] = top_by_field.get(group_field_value, []) + [rating]

            if metric == 'average':
                top_by_field = {k: Aux.list_average(v) for k, v in top_by_field.items()}
            elif metric == 'median':
                top_by_field = {k: Aux.list_median(v) for k, v in top_by_field.items()}

            if self.group_field == 1:
                top_by_field = {self.outer._get_title(k): v for k, v in top_by_field.items()}

            top_by_field = {k: round(v, 2) for k,v in sorted(top_by_field.items(), key=lambda x: x[1], reverse=True)[:n]}

            return top_by_field

        def top_controversial(self, n):
            ratings_csv_generator = Aux.csv_to_generator(self.outer.ratings_file, self.outer.rows_count)

            top_by_variance = dict()
            for line in ratings_csv_generator:
                parsed_fields = Aux.parse_csv_line(line)
                group_field_value = parsed_fields[self.group_field]
                rating = float(parsed_fields[2])
                top_by_variance[group_field_value] = top_by_variance.get(group_field_value, []) + [rating]

            top_by_variance = {k: Aux.list_variance(v) for k, v in top_by_variance.items()}

            if self.group_field == 1:
                top_by_variance = {self.outer._get_title(k): v for k, v in top_by_variance.items()}

            top_by_variance = {k: round(v, 2) for k,v in sorted(top_by_variance.items(), key=lambda x: x[1], reverse=True)[:n]}

            return top_by_variance

    class Users(Movies):

        def __init__(self, Ratings_instance):
            self.outer = Ratings_instance
            self.group_field = 0

    def _get_title(self, search_movie_id):
        with open(self.movies_file, 'r') as file_in:
            for line in file_in:
                line = line.strip()
                movie_id, title, _ = Aux.parse_csv_line(line)
                if movie_id == search_movie_id:
                    return title

    def _get_movie_ratings(self, movie_id):
        if not hasattr(self, '_movie_ratings_cache'):
            self._movie_ratings_cache = defaultdict(list)
            ratings_csv_generator = Aux.csv_to_generator(self.ratings_file, self.rows_count)

            for line in ratings_csv_generator:
                _, movie_id, rating, _ = Aux.parse_csv_line(line)
                self._movie_ratings_cache[movie_id].append(float(rating))

        return self._movie_ratings_cache.get(movie_id, [])

    def most_controversial_users_favorite_movies(self, n_users=10, n_movies=10):
        """
        self explanatory name
        """
        controversial_users = self.top_controversial_users(n_users)

        user_ratings = defaultdict(list)
        movie_ratings = defaultdict(list)

        ratings_csv_generator = Aux.csv_to_generator(self.ratings_file, self.rows_count)

        for line in ratings_csv_generator:
            user_id, movie_id, rating, _ = Aux.parse_csv_line(line)
            if user_id in controversial_users:
                user_ratings[user_id].append((movie_id, float(rating)))
                movie_ratings[movie_id].append(float(rating))

        movie_avg_ratings = {}
        for movie_id, ratings in movie_ratings.items():
            if len(ratings) >= 2:
                movie_avg_ratings[movie_id] = sum(ratings) / len(ratings)

        movies_with_titles = {}
        for movie_id, avg_rating in movie_avg_ratings.items():
            title = self._get_title(movie_id)
            if title:
                movies_with_titles[title] = round(avg_rating, 2)

        return dict(sorted(movies_with_titles.items(),
                        key=lambda x: x[1], reverse=True)[:n_movies])

    def most_controversial_users_hated_movies(self, n_users=10, n_movies=10):
        """
        also self explanatory, opposite of controversial users' favorite movies
        """
        controversial_users = self.top_controversial_users(n_users)

        movie_ratings = defaultdict(list)

        ratings_csv_generator = Aux.csv_to_generator(self.ratings_file, self.rows_count)

        for line in ratings_csv_generator:
            user_id, movie_id, rating, _ = Aux.parse_csv_line(line)
            if user_id in controversial_users:
                movie_ratings[movie_id].append(float(rating))

        movie_avg_ratings = {}
        for movie_id, ratings in movie_ratings.items():
            if len(ratings) >= 2:
                movie_avg_ratings[movie_id] = sum(ratings) / len(ratings)

        movies_with_titles = {}
        for movie_id, avg_rating in movie_avg_ratings.items():
            title = self._get_title(movie_id)
            if title:
                movies_with_titles[title] = round(avg_rating, 2)

        return dict(sorted(movies_with_titles.items(),
                        key=lambda x: x[1])[:n_movies])


class Links:
    """
    analyzing links.csv
    """
    def __init__(self, path_to_the_file, rows_count=1000):
        self.links_file = path_to_the_file
        self.rows_count = rows_count
        self.preprocessed = self._preprocess()
        self.preprocessed_fields = {'title': 0, 'director': 1, 'budget': 2, 'cumulative worldwide gross': 3, 'runtime': 4}

    def get_imdb(self, list_of_movies, list_of_fields=['movie id', 'title', 'director', 'budget', 'cumulative worldwide gross', 'runtime']):
        """
        a list of lists [movieId, field1, field2, field3, ...] for the list of movies given as the argument (movieId).
        For example, [movie id, Title, Director, Budget, Cumulative Worldwide Gross, Runtime].
        """
        imdb_info = []
        if list_of_fields[0].lower() != 'movie id':
            list_of_fields = ['movie id'] + list_of_fields

        for movie_id in list_of_movies:
            info = [None for _ in range(len(list_of_fields))]
            info[0] = movie_id
            if movie_id in self.preprocessed:
                for i, field in enumerate(list_of_fields):
                    if field in self.preprocessed_fields:
                        info[i] = self.preprocessed[movie_id][self.preprocessed_fields[field]]
            imdb_info.append(info)

        imdb_info.sort(key=lambda x: int(x[0]), reverse=True)

        return imdb_info


    def top_directors(self, n=None):
        """
        top directors by number of movies created
        """
        director_field = self.preprocessed_fields['director']

        directors = Counter((info[director_field] for info in self.preprocessed.values() if info[director_field]))
        n = (self.rows_count if n is None else n)
        directors = dict(directors.most_common(n))

        return directors


    def most_expensive(self, n=None):
        """
        top movies by budget
        """
        title_field = self.preprocessed_fields['title']
        budget_field = self.preprocessed_fields['budget']

        budgets = [(info[title_field], info[budget_field]) for info in self.preprocessed.values() if all((info[title_field], info[budget_field]))]
        n = (self.rows_count if n is None else n)
        budgets = dict(sorted(budgets, key=lambda x: x[1], reverse=True)[:n])

        return budgets


    def most_profitable(self, n=None):
        """
        top movies by difference between cumulative worldwide gross and budget
        """
        title_field = self.preprocessed_fields['title']
        budget_field = self.preprocessed_fields['budget']
        gross_field = self.preprocessed_fields['cumulative worldwide gross']

        profits = [(info[title_field], info[gross_field] - info[budget_field]) for info in self.preprocessed.values() if all((info[title_field],  info[gross_field], info[budget_field]))]
        n = (self.rows_count if n is None else n)
        profits = dict(sorted(profits, key=lambda x: x[1], reverse=True)[:n])

        return profits

    def longest(self, n=None):
        """
        top movies by runtime
        """
        title_field = self.preprocessed_fields['title']
        runtime_field = self.preprocessed_fields['runtime']

        runtimes = [(info[title_field], info[runtime_field]) for info in self.preprocessed.values() if all((info[title_field], info[runtime_field]))]
        n = (self.rows_count if n is None else n)
        runtimes = dict(sorted(runtimes, key=lambda x: x[1], reverse=True)[:n])

        return runtimes

    def top_cost_per_minute(self, n=None):
        """
        top movies by budget divided by runtime (costs)
        """
        title_field = self.preprocessed_fields['title']
        budget_field = self.preprocessed_fields['budget']
        runtime_field = self.preprocessed_fields['runtime']

        costs = [(info[title_field], round(info[budget_field] / info[runtime_field], 2)) for info in self.preprocessed.values() if all((info[title_field], info[budget_field], info[runtime_field]))]
        n = (self.rows_count if n is None else n)
        costs = dict(sorted(costs, key=lambda x: x[1], reverse=True)[:n])

        return costs

    def director_with_biggest_cumulative_budget(self):
        """
        finding a director whose movies have the biggest cumulative budget
        """
        director_budgets = defaultdict(int)
        director_field = self.preprocessed_fields['director']
        budget_field = self.preprocessed_fields['budget']

        for movie_info in self.preprocessed.values():
            director = movie_info[director_field]
            budget = movie_info[budget_field]

            if director and budget:
                directors = [d.strip() for d in director.split(',') if d.strip()]
                for single_director in directors:
                    director_budgets[single_director] += budget

        if not director_budgets:
            return {}
        max_director = max(director_budgets.items(), key=lambda x: x[1])
        return {max_director[0]: max_director[1]}

    def directors_cumulative_budgets(self, n=None):
        """
        helpful to find out the total money spent on movies by a director
        """
        director_budgets = defaultdict(int)
        director_field = self.preprocessed_fields['director']
        budget_field = self.preprocessed_fields['budget']

        for movie_info in self.preprocessed.values():
            director = movie_info[director_field]
            budget = movie_info[budget_field]

            if director and budget:
                directors = [d.strip() for d in director.split(',') if d.strip()]
                for single_director in directors:
                    director_budgets[single_director] += budget

        n = self.rows_count if n is None else n
        return dict(sorted(director_budgets.items(),
                        key=lambda x: x[1], reverse=True)[:n])


    def _get_imdb(self, movie_id):
        with open(self.links_file, 'r') as file_in:
            for line in file_in:
                line = line.strip()
                current_movie_id, current_imdb, _ = Aux.parse_csv_line(line)
                if current_movie_id == movie_id:
                    return current_imdb

    def _get_soup_text(self, imdb):
        url = f'https://www.imdb.com/title/tt{imdb}/'
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/html"}
        r = requests.get(url=url, headers=headers)
        return BeautifulSoup(r.text, "html.parser")

    def _get_director(self, soup_text):
        try:
            name = soup_text.find("span", string="Director").next_sibling.text
        except Exception as e:
            name = None
        return name

    def _get_budget(self, soup_text):
        try:
            string = soup_text.find("span", string="Budget").next_sibling.text
            result = int(''.join([ch for ch in string if ch.isdigit()]))
        except Exception as e:
            result = None
        return result

    def _get_gross_worldwide(self, soup_text):
        try:
            string = soup_text.find("span", string="Gross worldwide").next_sibling.text
            result = int(''.join([ch for ch in string if ch.isdigit()]))
        except Exception as e:
            result = None
        return result

    def _get_runtime(self, soup_text):
        try:
            string = soup_text.find("span", string="Runtime").next_sibling.text
            result = int(re.search(r'\((\d+)\s*min\)', string).group(1))
        except Exception as e:
            result = None
        return result

    def _get_title(self, soup_text):
        try:
            result = soup_text.find("div", attrs={"class": "sc-cb6a22b2-1 kEdApw baseAlt"}).text
            if result is not None:
                result = result[result.find(':') + 2:]
            else:
                result = soup_text.find("span", attrs={"class": "hero__primary-text"}).text
        except Exception as e:
            result = None
        return result

    def _preprocess(self):
        result = dict()

        links_csv_generator = Aux.csv_to_generator(self.links_file, self.rows_count)

        for line in links_csv_generator:
            movie_id, imdb_id, _ = line.split(',')
            soup_text = self._get_soup_text(imdb_id)

            movie_info = []
            movie_info.append(self._get_title(soup_text))
            movie_info.append(self._get_director(soup_text))
            movie_info.append(self._get_budget(soup_text))
            movie_info.append(self._get_gross_worldwide(soup_text))
            movie_info.append(self._get_runtime(soup_text))

            result[movie_id] = movie_info

        return result



class Aux:

    def csv_to_generator(path_to_the_file, rows_count):
        with open(path_to_the_file, 'r') as file_in:
            for i, line in enumerate(file_in):
                if i > rows_count:
                    break
                if i != 0:
                    yield line


    def parse_csv_line(line):
        fields = []
        current_field = []
        in_quotes = False
        i = 0
        n = len(line)

        while i < n:
            char = line[i]

            if not in_quotes:
                if char == '"':
                    in_quotes = True
                    i += 1
                elif char == ',':
                    fields.append(''.join(current_field))
                    current_field = []
                    i += 1
                else:
                    current_field.append(char)
                    i += 1
            else:
                if char == '"':
                    if i + 1 < n and line[i + 1] == '"':
                        current_field.append('"')
                        i += 2
                    else:
                        in_quotes = False
                        i += 1
                else:
                    current_field.append(char)
                    i += 1

        fields.append(''.join(current_field))
        return fields

    def list_average(numbers):
        return sum(numbers) / len(numbers)

    def list_median(numbers):
        if len(numbers) % 2 == 0:
            median = (numbers[len(numbers) // 2 - 1] + numbers[len(numbers) // 2]) / 2
        else:
            median = numbers[len(numbers) // 2]
        return median

    def list_variance(numbers):
        avg = Aux.list_average(numbers)
        return sum((num - avg) ** 2 for num in numbers) / len(numbers)
    
class Test:
    """test all classes including Aux"""
    def test_csv_to_generator(self):
        generator = Aux.csv_to_generator('movies.csv', rows_count = 1)
        lines = list(generator)
        assert len(lines) == 1

    def test_parse_csv_line_normal(self):
        assert Aux.parse_csv_line('a,b,c') == ['a', 'b', 'c']

    def test_parse_csv_line_quotes(self):
        assert Aux.parse_csv_line('a,"b",c') == ['a', 'b', 'c']

    def test_list_average(self):
        assert Aux.list_average([1, 2, 3]) == 2

    def test_list_median_odd(self):
        assert Aux.list_median([1, 2, 3, 4, 5]) == 3

    def test_list_median_odd(self):
        assert Aux.list_median([1, 2, 3, 4]) == 2.5

    def test_list_variance(self):
        assert Aux.list_variance([1, 1, 1, 1]) == 0

    def test_list_variance_two(self):
        assert Aux.list_variance([1, 2, 3, 4, 5]) == 2

    def test_mov_init(self):
        movies = Movies('movies.csv')
        assert movies.file == 'movies.csv'
        assert hasattr(movies, 'rows_count')

    def test_dist_by_release_year(self):
        movies = Movies('movies.csv')
        result = movies.dist_by_release_year()
        assert isinstance(result, dict)
        assert all(isinstance(year, str) for year in result.keys())
        assert all(isinstance(count, int) for count in result.values())
        assert all(count > 0 for count in result.values())

    def test_dist_by_genres(self):
        movies = Movies('movies.csv')
        result = movies.dist_by_genres()
        assert isinstance(result, dict)
        assert all(isinstance(genre, str) for genre in result.keys())
        assert all(isinstance(count, int) for count in result.values())
        assert all(count > 0 for count in result.values())

    def test_most_genres(self):
        movies = Movies('movies.csv')
        result = movies.most_genres(5)
        assert isinstance(result, dict)
        assert len(result) <= 5
        assert all(isinstance(movie, str) for movie in result.keys())
        assert all(isinstance(count, int) for count in result.values())

    def test_tag_init(self):
        tags = Tags('tags.csv')
        assert tags.file == 'tags.csv'
        assert hasattr(tags, 'rows_count')

    def test_most_words(self):
        tags = Tags('tags.csv')
        result = tags.most_words(5)

        assert isinstance(result, dict)
        assert len(result) <= 5
        assert all(isinstance(tag, str) for tag in result.keys())
        assert all(isinstance(count, int) for count in result.values())

    def test_longest(self):
        tags = Tags('tags.csv')
        result = tags.longest(5)

        assert isinstance(result, list)
        assert len(result) <= 5
        assert all(isinstance(tag, str) for tag in result)

    def test_most_popular(self):
        tags = Tags('tags.csv')
        result = tags.most_popular(5)

        assert isinstance(result, dict)
        assert len(result) <= 5
        assert all(isinstance(tag, str) for tag in result.keys())
        assert all(isinstance(count, int) for count in result.values())

    def test_tags_with(self):
        tags = Tags('tags.csv')
        result = tags.tags_with('action')

        assert isinstance(result, list)
        assert all(isinstance(tag, str) for tag in result)

    def test_rate_init(self):
        ratings = Ratings('ratings.csv', 'movies.csv')
        assert ratings.ratings_file == 'ratings.csv'
        assert ratings.movies_file == 'movies.csv'
        assert hasattr(ratings, 'Movies_analyzer')
        assert hasattr(ratings, 'Users_analyzer')

    def test_ratings_dist_by_year(self):
        ratings = Ratings('ratings.csv', 'movies.csv')
        result = ratings.ratings_dist_by_year()

        assert isinstance(result, dict)
        assert all(isinstance(year, int) for year in result.keys())
        assert all(isinstance(count, int) for count in result.values())

    def test_ratings_dist_by_value(self):
        ratings = Ratings('ratings.csv', 'movies.csv')
        result = ratings.ratings_dist_by_value()

        assert isinstance(result, dict)
        assert all(isinstance(rating, str) for rating in result.keys())
        assert all(isinstance(count, int) for count in result.values())

    def test_top_movies_by_num_of_ratings(self):
        ratings = Ratings('ratings.csv', 'movies.csv')
        result = ratings.top_movies_by_num_of_ratings(5)

        assert isinstance(result, dict)
        assert len(result) <= 5
        assert all(isinstance(movie, str) for movie in result.keys())
        assert all(isinstance(count, int) for count in result.values())

    def test_top_movies_by_average_rating(self):
        ratings = Ratings('ratings.csv', 'movies.csv')
        result = ratings.top_movies_by_average_rating(5)

        assert isinstance(result, dict)
        assert len(result) <= 5
        assert all(isinstance(movie, str) for movie in result.keys())
        assert all(isinstance(rating, float) for rating in result.values())

    def test_most_controversial_users_favorite_movies(self):
        ratings = Ratings('ratings.csv', 'movies.csv')
        result = ratings.most_controversial_users_favorite_movies(n_users=3, n_movies=3)

        assert isinstance(result, dict)
        assert all(isinstance(movie, str) for movie in result.keys())
        assert all(isinstance(rating, float) for rating in result.values())

    def test_most_controversial_users_hated_movies(self):
        ratings = Ratings('ratings.csv', 'movies.csv')
        result = ratings.most_controversial_users_hated_movies(n_users=3, n_movies=3)

        assert isinstance(result, dict)
        assert all(isinstance(movie, str) for movie in result.keys())
        assert all(isinstance(rating, float) for rating in result.values())

    def test_link_init(self):
        links = Links('links.csv', rows_count=10)
        assert links.links_file == 'links.csv'
        assert hasattr(links, 'preprocessed')
        assert hasattr(links, 'preprocessed_fields')

    def test_get_imdb(self):
        links = Links('links.csv', rows_count=10)
        
        result = links.get_imdb(['1', '2'], ['title', 'director'])
        
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, list)
            assert len(item) == 3

    def test_top_directors(self):
        links = Links('links.csv', rows_count=10)
        result = links.top_directors(5)
        
        assert isinstance(result, dict)
        assert len(result) <= 5
        assert all(isinstance(director, str) for director in result.keys())
        assert all(isinstance(count, int) for count in result.values())

    def test_most_expensive(self):
        links = Links('links.csv', rows_count=10)
        result = links.most_expensive(5)
        
        assert isinstance(result, dict)
        assert len(result) <= 5
        assert all(isinstance(movie, str) for movie in result.keys())
        assert all(isinstance(budget, int) for budget in result.values())

    def test_most_profitable(self):
        links = Links('links.csv', rows_count=10)
        result = links.most_profitable(5)
        
        assert isinstance(result, dict)
        assert len(result) <= 5
        assert all(isinstance(movie, str) for movie in result.keys())
        assert all(isinstance(profit, int) for profit in result.values())

    def test_director_with_biggest_cumulative_budget(self):
        links = Links('links.csv', rows_count=10)
        result = links.director_with_biggest_cumulative_budget()
        
        assert isinstance(result, dict)
        assert len(result) == 1
        assert all(isinstance(director, str) for director in result.keys())
        assert all(isinstance(budget, int) for budget in result.values())