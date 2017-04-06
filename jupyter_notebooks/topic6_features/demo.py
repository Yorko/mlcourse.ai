import numpy as np
import pandas as pd
import json
from sklearn.base import TransformerMixin

EPSILON = 1e-5


class FeatureEngineer(TransformerMixin):

    def apply(self, df, k, condition):
        df[k] = df['features'].apply(condition)
        df[k] = df[k].astype(np.int8)

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None):
        df = X.copy()

        df.features = df.features.apply(lambda x: ' '.join([y.replace(' ', '_') for y in x]))
        df.features = df.features.apply(lambda x: x.lower())
        df.features = df.features.apply(lambda x: x.replace('-', '_'))

        for k, condition in (('dishwasher', lambda x: 'dishwasher' in x),
                             ('doorman', lambda x: 'doorman' in x or 'concierge' in x),
                             ('pets', lambda x: "pets" in x or "pet" in x or "dog" in x or "cats" in x and "no_pets" not in x),
                             ('air_conditioning', lambda x: 'air_conditioning' in x or 'central' in x),
                             ('parking', lambda x: 'parking' in x),
                             ('balcony', lambda x: 'balcony' in x or 'deck' in x or 'terrace' in x or 'patio' in x),
                             ('bike', lambda x: 'bike' in x),
                             ('storage', lambda x: 'storage' in x),
                             ('outdoor', lambda x: 'outdoor' in x or 'courtyard' in x or 'garden' in x),
                             ('roof', lambda x: 'roof' in x),
                             ('gym', lambda x: 'gym' in x or 'fitness' in x),
                             ('pool', lambda x: 'pool' in x),
                             ('backyard', lambda x: 'backyard' in x),
                             ('laundry', lambda x: 'laundry' in x),
                             ('hardwood_floors', lambda x: 'hardwood_floors' in x),
                             ('new_construction', lambda x: 'new_construction' in x),
                             ('dryer', lambda x: 'dryer' in x),
                             ('elevator', lambda x: 'elevator' in x),
                             ('garage', lambda x: 'garage' in x),
                             ('pre_war', lambda x: 'pre_war' in x or 'prewar' in x),
                             ('post_war', lambda x: 'post_war' in x or 'postwar' in x),
                             ('no_fee', lambda x: 'no_fee' in x),
                             ('low_fee', lambda x: 'reduced_fee' in x or 'low_fee' in x),
                             ('fire', lambda x: 'fireplace' in x),
                             ('private', lambda x: 'private' in x),
                             ('wheelchair', lambda x: 'wheelchair' in x),
                             ('internet', lambda x: 'wifi' in x or 'wi_fi' in x or 'internet' in x),
                             ('yoga', lambda x: 'yoga' in x),
                             ('furnished', lambda x: 'furnished' in x),
                             ('multi_level', lambda x: 'multi_level' in x),
                             ('exclusive', lambda x: 'exclusive' in x),
                             ('high_ceil', lambda x: 'high_ceil' in x),
                             ('green', lambda x: 'green_b' in x),
                             ('stainless', lambda x: 'stainless_' in x),
                             ('simplex', lambda x: 'simplex' in x),
                             ('public', lambda x: 'public' in x),
                             ):
            self.apply(df, k, condition)

        df['bathrooms'] = df['bathrooms'].apply(lambda x: x if x < 5 else 5)
        df['bedrooms'] = df['bedrooms'].apply(lambda x: x if x < 5 else 5)
        df["num_photos"] = df["photos"].apply(len)
        df["num_features"] = df["features"].apply(len)
        created = pd.to_datetime(df.pop("created"))
        df["listing_age"] = (pd.to_datetime('today') - created).apply(lambda x: x.days)
        df["room_dif"] = df["bedrooms"] - df["bathrooms"]
        df["room_sum"] = df["bedrooms"] + df["bathrooms"]
        df["price_per_room"] = df["price"] / df["room_sum"].apply(lambda x: max(x, .5))
        df["bedrooms_share"] = df["bedrooms"] / df["room_sum"].apply(lambda x: max(x, .5))
        df['price'] = df['price'].apply(lambda x: np.log(x + EPSILON))

        key_types = df.dtypes.to_dict()
        for k in key_types:
            if key_types[k].name not in ('int64', 'float64', 'int8'):
                df.pop(k)

        for k in ('latitude', 'longitude', 'listing_id'):
            df.pop(k)
        return df


def encode(x):
    if x == 'low':
        return 0
    elif x == 'medium':
        return 1
    elif x == 'high':
        return 2


def get_data():
    with open('train.json', 'r') as raw_data:
        data = json.load(raw_data)

    df = pd.DataFrame(data)
    target = df.pop('interest_level').apply(encode)

    df = FeatureEngineer().fit_transform(df)
    return df, target
