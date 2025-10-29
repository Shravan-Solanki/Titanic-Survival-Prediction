import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train['Title'] = train['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
test['Title'] = test['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
train['Title'] = train['Title'].replace({'Mlle': 'Miss',
                                   'Ms': 'Miss',
                                   'Mme': 'Mrs'})
test['Title'] = test['Title'].replace({'Mlle': 'Miss',
                                   'Ms': 'Miss',
                                   'Mme': 'Mrs'})
rare_titles = ['Lady', 'Countess', 'Capt', 'Don', 'Sir', 'Jonkheer']

train['Title'] = train['Title'].replace(rare_titles, 'Rare')
test['Title'] = test['Title'].replace(rare_titles, 'Rare')

train['TicketPrefix'] = train['Ticket'].str.extract(r'([A-Za-z\.\/]+)')
test['TicketPrefix'] = test['Ticket'].str.extract(r'([A-Za-z\.\/]+)')
train['TicketPrefix'] = train['TicketPrefix'].fillna('NoPrefix')
test['TicketPrefix'] = test['TicketPrefix'].fillna('NoPrefix')
train['TicketPrefix'] = train['TicketPrefix'].apply(lambda x:x if train['TicketPrefix'].value_counts()[x] > 1 else 'Rare')
test['TicketPrefix'] = test['TicketPrefix'].apply(lambda x:x if test['TicketPrefix'].value_counts()[x] > 1 else 'Rare')

ticket_counts = train['Ticket'].value_counts()
ticket_counts1 = test['Ticket'].value_counts()
test['TicketGroupSize'] = test['Ticket'].map(ticket_counts1)
train['TicketGroupSize'] = train['Ticket'].map(ticket_counts)

train['Deck'] = train['Cabin'].str[0]
test['Deck'] = test['Cabin'].str[0]

p_id = test['PassengerId']
test.drop(['PassengerId','Ticket','Cabin','Name'],axis=1,inplace=True)
train.drop(['PassengerId','Ticket','Cabin','Name'],axis=1,inplace=True)

train = pd.get_dummies(train,columns=['Sex','Title','TicketPrefix'],drop_first=True,dtype=int)
test = pd.get_dummies(test,columns=['Sex','Title','TicketPrefix'],drop_first=True,dtype=int)

le = LabelEncoder()
l1 = LabelEncoder()

non_null_deck = train['Deck'][train['Deck'].notnull()]
non_null_em = train['Embarked'][train['Embarked'].notnull()]

le.fit(non_null_deck)
l1.fit(non_null_em)

train['Deck'] = train['Deck'].apply(lambda x: le.transform([x])[0] if pd.notnull(x) else np.nan)
test['Deck'] = test['Deck'].apply(lambda x: le.transform([x])[0] if pd.notnull(x) else np.nan)
train['Embarked'] = train['Embarked'].apply(lambda x: l1.transform([x])[0] if pd.notnull(x) else np.nan)
test['Embarked'] = test['Embarked'].apply(lambda x: l1.transform([x])[0] if pd.notnull(x) else np.nan)
feature = train.columns.tolist()
feature.remove('Survived')
feature1 = test.columns.tolist()

imputer = KNNImputer(n_neighbors=13)

train[feature] = imputer.fit_transform(train[feature])
test[feature1] = imputer.fit_transform(test[feature1])
train['Deck'] = train['Deck'].round().astype(int)
test['Deck'] = test['Deck'].round().astype(int)
train['Embarked'] = train['Embarked'].round().astype(int)
test['Embarked'] = test['Embarked'].round().astype(int)

train['Deck'] = train['Deck'].apply(lambda x: le.classes_[x])
test['Deck'] = test['Deck'].apply(lambda x: le.classes_[x])
train['Embarked'] = train['Embarked'].apply(lambda x: l1.classes_[x])
test['Embarked'] = test['Embarked'].apply(lambda x: l1.classes_[x])

train = pd.get_dummies(train,columns=['Deck','Embarked'],drop_first=True,dtype=float)
test = pd.get_dummies(test,columns=['Deck','Embarked'],drop_first=True,dtype=float)

train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
train['IsAlone'] = 0
test['IsAlone'] = 0
train.loc[train['FamilySize'] == 1, 'IsAlone'] = 1
test.loc[test['FamilySize'] == 1, 'IsAlone'] = 1

X_train = train.drop('Survived', axis=1)
y_train = train['Survived']

test = test.reindex(columns=X_train.columns, fill_value=0)

scaler = StandardScaler()
X_train[['Age','Fare']] = scaler.fit_transform(X_train[['Age','Fare']])
test[['Age','Fare']] = scaler.transform(test[['Age','Fare']])

# Random Forest
forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(X_train, y_train)

predictions = forest_clf.predict(test).astype(int)

submission = pd.DataFrame({
    "PassengerId": p_id,
    "Survived": predictions
})

submission.to_csv("titanic_predictions.csv", index=False)

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

forest_clf.fit(X_tr, y_tr)
val_pred = forest_clf.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, val_pred))

