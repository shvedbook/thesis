import uvicorn

import pickle

from fastapi import FastAPI
from xgboost import XGBRegressor
from pydantic import BaseModel,Field
import pandas as pd

feature_names = {'feature_1': 'מיתווה או תבנית (פורמט)',
 'feature_2': 'שמירת סדר הקודקס: שומרי  דפים/גיליונות',
 'feature_3': 'סימני מים:- יש-',
 'feature_4': 'ניהול שורה: יישור וחריגה',
 'feature_5': 'תיבות פתיחה או כותרות:- יש-',
 'feature_6': 'השורה התחתונה:- לא בתוך המסגרת-',
 'feature_7': 'קלף',
 'feature_8': 'שרטוט בחריטה: גיליון או דף- גיליון גיליון:-',
 'feature_9': 'לשון הטקסט (חוץ מעברית):- ערבית-',
 'feature_10': 'התחלת המילה הבאה:- עם סימן-',
 'feature_11': 'אמצעים לשמירת רצף הקודקס (כללי)',
 'feature_12': 'שומר קונטרס:- מאוזן-',
 'feature_13': 'כותרת רצה וסימון מיפתח אמצע קונטרס',
 'feature_14': 'מבריחים:- מבריחים יחידים-',
 'feature_15': 'תנ"ך:',
 'feature_16': 'קונטרס מתחיל בצד:- שיער-',
 'feature_17': 'מבריחים:- מבריחים לא נראים-',
 'feature_18': 'שמירת סדר הקודקס: שומרי קונטרסים',
 'feature_19': 'תנ"ך:- טקסט-',
 'feature_20': 'מיתווה או תבנית (פורמט)- אחר:-',
 'feature_21': 'קונטרס מתחיל בצד:- בשר-',
 'feature_22': 'מילוי שורה (כללי) חריגה או מניעתה:- יש-',
 'feature_23': 'שרטוט בחריטה: 2 גיליונות או 2 דפים- שני דפים בבת אחת כסדרם בקונטרס:-',
 'feature_24': 'סימני מים:- אין-',
 'feature_25': 'קישוטים / שקיפות הטקסט- קישוטים:-',
 'feature_26': 'מצע הכתיבה: קלף ונייר',
 'feature_27': 'הלכה ומדרש:',
 'feature_28': 'קווים:- קווים נראים-',
 'feature_29': 'מילוי גרפי:- בסוף השורה-',
 'feature_30': 'ניהול שורה: מניעת חריגה: ריווח וציפוף',
 'feature_31': 'צירופי אותיות- אותיות בתוך אותיות-',
 'feature_32': 'שומר דף:- מלוכסן-',
 'feature_33': 'דיו בצבע:- חום כהה-',
 'feature_34': 'תפילה ומחזורים:',
 'feature_35': 'מבריק:- כל הדף-',
 'feature_36': 'קישוטים:- אין-',
 'feature_37': "דף דף:- בצד עמוד ב'-",
 'feature_38': 'משאלות בשוליים תחתונים',
 'feature_39': 'דיו בצבע:- חום-',
 'feature_40': 'מבריחים בקבוצות:- בקבוצות של 3-',
 'feature_41': 'ניהול שורה: מילוי שורה',
 'feature_42': 'מצע הכתיבה: נייר- קווים:-',
 'feature_43': 'זוג נקבים לשרטוט למלוא רוחב העמוד',
 'feature_44': 'דיו בצבע:- שחור-',
 'feature_45': 'פילוסופיה וקבלה:',
 'feature_46': 'שרטוט בחריטה (כללי)',
 'feature_47': 'גיליון גיליון:- פרוש-',
 'feature_48': 'קישוטים / שקיפות הטקסט- תיבות פתיחה או כותרות:-',
 'feature_49': 'ספרור קונטרסים:- בראש--',
 'feature_50': 'מתפצל או פציל:- 2 שכבות-'}


class Codicological_Data(BaseModel):
    feature_1: int =  Field (ge=0, le=1)
    feature_2: int =  Field (ge=0, le=1)
    feature_3: int =  Field (ge=0, le=1)
    feature_4: int =  Field (ge=0, le=1)
    feature_5: int =  Field (ge=0, le=1)
    feature_6: int =  Field (ge=0, le=1)
    feature_7: int =  Field (ge=0, le=1)
    feature_8: int =  Field (ge=0, le=1)
    feature_9: int =  Field (ge=0, le=1)
    feature_10: int =  Field (ge=0, le=1)
    feature_11: int =  Field (ge=0, le=1)
    feature_12: int =  Field (ge=0, le=1)
    feature_13: int =  Field (ge=0, le=1)
    feature_14: int =  Field (ge=0, le=1)
    feature_15: int =  Field (ge=0, le=1)
    feature_16: int =  Field (ge=0, le=1)
    feature_17: int =  Field (ge=0, le=1)
    feature_18: int =  Field (ge=0, le=1)
    feature_19: int =  Field (ge=0, le=1)
    feature_20: int =  Field (ge=0, le=1)
    feature_21: int =  Field (ge=0, le=1)
    feature_22: int =  Field (ge=0, le=1)
    feature_23: int =  Field (ge=0, le=1)
    feature_24: int =  Field (ge=0, le=1)
    feature_25: int =  Field (ge=0, le=1)
    feature_26: int =  Field (ge=0, le=1)
    feature_27: int =  Field (ge=0, le=1)
    feature_28: int =  Field (ge=0, le=1)
    feature_29: int =  Field (ge=0, le=1)
    feature_30: int =  Field (ge=0, le=1)
    feature_31: int =  Field (ge=0, le=1)
    feature_32: int =  Field (ge=0, le=1)
    feature_33: int =  Field (ge=0, le=1)
    feature_34: int =  Field (ge=0, le=1)
    feature_35: int =  Field (ge=0, le=1)
    feature_36: int =  Field (ge=0, le=1)
    feature_37: int =  Field (ge=0, le=1)
    feature_38: int =  Field (ge=0, le=1)
    feature_39: int =  Field (ge=0, le=1)
    feature_40: int =  Field (ge=0, le=1)
    feature_41: int =  Field (ge=0, le=1)
    feature_42: int =  Field (ge=0, le=1)
    feature_43: int =  Field (ge=0, le=1)
    feature_44: int =  Field (ge=0, le=1)
    feature_45: int =  Field (ge=0, le=1)
    feature_46: int =  Field (ge=0, le=1)
    feature_47: int =  Field (ge=0, le=1)
    feature_48: int =  Field (ge=0, le=1)
    feature_49: int =  Field (ge=0, le=1)
    feature_50: int =  Field (ge=0, le=1)

app = FastAPI()




model = XGBRegressor()
model.load_model("FastAPI Files/model.txt")

@app.get('/')

def index():

    return {'message': 'This is the homepage of the API '}

@app.post('/prediction')

def predict_production_year(data: Codicological_Data):
    received = data.dict()
    df_dict = {}

    for k in received:
        df_dict[feature_names[k]] = list(str(received[k]))
    df = (pd.DataFrame.from_dict(df_dict)).astype(float)
    prediction = model.predict(df)
    print(prediction)
    return {'prediction': str(prediction[0])}

if __name__ == '__main__':

    uvicorn.run(app, host='127.0.0.1', port=5000, debug=True)