import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# ============================================================================
# SETUP & PATHS (Exact Match to Project Structure)
# ============================================================================

# Current script location: AI/scripts/domain_analysis.py
SCRIPT_DIR = Path(__file__).resolve().parent

# Root folder: AI/
PROJECT_ROOT = SCRIPT_DIR.parent 

# Data directory: AI/data/
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


# ============================================================================
# COMPREHENSIVE AMHARIC-ONLY DOMAIN KEYWORDS (Expanded for High Coverage)
# ============================================================================

DOMAIN_KEYWORDS = {
 'Government': [
        # General, Administrative & Public Service
        'መንግስት', 'መንግሥት', 'ሥርዓት', 'አስተዳደር', 'ከተማ አስተዳደር', 'የክልል መንግስት', 
        'ፌዴራል', 'ፌዴራላዊ', 'ሪፐብሊክ', 'ክልል', 'ወረዳ', 'ቀበሌ', 'መዋቅር', 'ቢሮ', 
        'ጽህፈት ቤት', 'ጽሕፈት ቤት', 'ሴክተር', 'ኤጀንሲ', 'ኮሚሽን', 'ባለስልጣን', 'ስልጣን', 
        'መመሪያ', 'ደንብ', 'መግለጫ', 'ሪፖርት', 'ውሳኔ', 'ማስፈጸሚያ', 'ካቢኔ', 'ስቪል ሰርቪስ', 
        'የሲቪል ሰርቪስ ሚኒስቴር', 'የሲቪል ሰርቪስ ኮሚሽን', 'የህዝብ አገልግሎት', 'አስተዳደራዊ', 
        'ሹመት', 'ሹማምንት', 'ስብሰባ', 'መድረክ', 'የውይይት መድረክ', 'ቃል አቀባይ', 'አማካሪ',

        # Executive & Leadership Roles
        'ጠቅላይ ሚኒስትር', 'ምክትል ጠቅላይ ሚኒስትር', 'ፕሬዝዳንት', 'ምክትል ፕሬዝዳንት', 
        'ፕሬዚዳንት', 'ሚኒስቴር', 'ሚኒስተር', 'ሚኒስትር ዴኤታ', 'ዋና ዳይሬክተር', 'ኮሚሽነር', 
        'ከንቲባ', 'ዋና ከንቲባ', 'ሊቀመንበር', 'ዋና አስተዳዳሪ', 'ምክትል አስተዳዳሪ', 
        'አምባሳደር', 'ዲፕሎማሲ', 'ፓስፖርት', 'ቪዛ',

        # Ministries & Specific Government Offices
        'የውጭ ጉዳይ ሚኒስቴር', 'የገንዘብ ሚኒስቴር', 'የሰላም ሚኒስቴር', 'የፍትህ ሚኒስቴር', 
        'የንግድ ሚኒስቴር', 'የግብርና ሚኒስቴር', 'የጤና ሚኒስቴር', 'የገቢዎች ሚኒስቴር', 
        'የፈጠራ እና ቴክኖሎጂ ሚኒስቴር', 'የሰነዶች ማረጋገጫ', 'የምርጫ ቦርድ',

        # Politics, Elections & Parties
        'ምርጫ', 'ፖለቲካ', 'ፖለቲከኛ', 'ፓርቲ', 'የፖለቲካ ፓርቲ', 'ተቃዋሚ', 'ገዢ ፓርቲ', 
        'ሕዝባዊ ሰልፍ', 'ድምጽ', 'ምርጫ ሰሌዳ', 'መራጭ', 'ታዛቢ', 'ዲሞክራሲ', 

        # Laws, Rights, Taxes & Financial Governance
        'ህግ', 'ሕግ', 'ሕግጋት', 'ህጎች', 'አዋጅ', 'ህገ መንግስት', 'ሕገ መንግሥት', 
        'ፖሊሲ', 'መብት', 'ግዴታ', 'ዜጋ', 'ዜግነት', 'ታክስ', 'ግብር', 'በጀት', 
        'መበጀት', 'በጀት ዓመት', 'ኦዲት', 'ሙስና', 'ሉዓላዊነት', 'ድንበር', 'ህገወጥ', 
        'ህጋዊ', 'የተከለከለ',

        # Legislative & Councils
        'ፓርላማ', 'ምክር ቤት', 'ቤት ተወካዮች', 'ተወካዮች', 'ፌዴሬሽን', 'ህግ አውጪ', 
        'ህግ አስፈፃሚ', 'መሪያ', 'የምክር ቤት ስብሰባ', 

        # Judiciary, Legal & Law Enforcement
        'ፍትህ', 'የፍትህ ስርዓት', 'ፍርድ ቤት', 'ጠቅላይ ፍርድ ቤት', 'ዳኛ', 'አቃቤ ህግ', 
        'እስር ቤት', 'ማረሚያ ቤት', 'ወንጀል', 'ምርመራ', 'ክስ', 'ከሳሽ', 'ተከሳሽ', 
        'ምስክር', 'የምስክር ቃል', 'የፍርድ ቤት ውሳኔ', 'ይግባኝ', 'የሰበር ሰሚ ችוט', 
        'የሰበር ችሎት', 'ቅጣት', 'የዕድሜ ልክ እስራት', 'የሞት ቅጣት', 'ማረሚያ', 
        'ማስተካከያ', 'ፖሊስ', 'ፖሊስ ጣቢያ', 'የፌዴራል ፖሊስ', 'የክልል ፖሊስ', 

        # Defense, Security & Peace
        'ጸጥታ', 'ሰላም', 'ሰራዊት', 'ሠራዊት', 'ሠራዊቱ', 'ወታደር', 'ጦር', 'መከላከያ', 
        'የመከላከያ ሰራዊት', 'አሸባሪ', 'ብሔራዊ ደህንነት', 'ደህንነት', 'ሠራዊት አዛዥ', 
        'ጄኔራል', 'ኮሎኔል', 'የጦር ሰፈር', 'ወታደራዊ', 'የጸጥታ ኃይል', 'ሰላም ማስከበር', 
        'ስልጠና', 'ጥቃት', 'ጥበቃ', 'ድንበር ጥበቃ'
    ],

   'Education': [
        # General & School Levels
        'ትምህርት', 'ትምህርት ቤት', 'ትምህርት ቤቶች', 'ትምህርትቤት', 'ትምህርቶች', 'ትምህርቱን', 
        'አንደኛ ደረጃ', 'ሁለተኛ ደረጃ', 'መሰናዶ', 'መዋለ ሕጻናት', 'ኪንደርጋርደን', 'ቴክኒክና ሙያ', 
        'አካዳሚ', 'ዩኒቨርሲቲ', 'ዩኒቨርሲቲዎች', 'ኮሌጅ', 'ኮሌጆች', 'ኮሌጆችና ዩኒቨርሲቲዎች', 
        'የትምህርት እድል', 'ስኮላርሺፕ', 'ትምህርት ቤት ውስጥ', 'የዩኒቨርሲቲ ተማሪ',

        # People & Academic Roles
        'ተማሪ', 'ተማሪዎች', 'መምህር', 'መምህራን', 'አስተማሪ', 'ፕሮፌሰር', 'ተባባሪ ፕሮፌሰር', 
        'ረዳት ፕሮፌሰር', 'ተመራማሪ', 'ዲን', 'ሬጅስትራር', 'የወላጆች ማህበር', 'የህጻናት አስተማሪ',

        # Materials & Classroom Supplies
        'መጽሐፍ', 'መጽሃፍ', 'የመማሪያ መጽሐፍ', 'መማሪያ', 'ክፍል', 'ደብተር', 'ማስታወሻ ደብተር', 
        'ማስታወሻ', 'ሰሌዳ', 'ኖራ', 'ማጥፊያ', 'እርሳስ', 'እስክሪብቶ', 'ዩኒፎርም', 'እረፍት', 
        'የክፍል ጊዜ', 'ቤተ መጻሕፍት',

        # Academics, Courses & Curriculum
        'መማር', 'ማስተማር', 'ትምህርታዊ', 'ኮርስ', 'ስርዓተ ትምህርት', 'መርሃ ትምህርት', 
        'መርሃ ግብር', 'ዲፓርትመንት', 'ፋኩልቲ', 'ዲግሪ', 'የመጀመሪያ ዲግሪ', 'ማስተርስ', 
        'ዶክተሬት', 'ዲፕሎማ', 'ሰርተፍኬት', 'የምስክር ወረቀት', 'ማዕረግ', 'የምርምር ስራ', 
        'የምርምር ማዕከል', 'ስልጠና', 'የትምህርት ጥራት',

        # Evaluation & Examinations
        'ፈተና', 'ፈተናዎች', 'ብሔራዊ ፈተና', 'ጥያቄ', 'መልስ', 'መልመጃ', 'ውጤት', 
        'ግምገማ', 'ጥናት',

        # Subjects & Disciplines
        'ሳይንስ', 'ሒሳብ', 'ታሪክ', 'ቋንቋ', 'ፊዚክስ', 'ኬሚስትሪ', 'ባዮሎጂ', 'ጂኦግራፊ', 
        'ሲቪክ', 'ኪነ ጥበብ', 'ሙዚቃ', 'ስፖርት', 'ኮምፒውተር', 'አይሲቲ', 'የኢንፎርሜሽን ቴክኖሎጂ',

        # Administration & Governance
        'የትምህርት ሚኒስቴር', 'ትምህርት ሚኒስቴር', 'የትምህርት ቢሮ', 'የወረዳ ትምህርት ጽህፈት ቤት', 
        'የትምህርት ቤት ቦርድ'
    ],

    'Religion': [
        'እግዚአብሔር', 'አምላክ', 'ጌታ', 'ኢየሱስ', 'ክርስቶስ', 'ቤተክርስቲያን', 'መስጊድ', 
        'ጸሎት', 'ፀሎት', 'ቅዳሴ', 'ሃይማኖት', 'ሃይማኖታዊ', 'እምነት', 'መጽሐፍ ቅዱስ', 
        'ቁርዓን', 'ቁርአን', 'ነቢይ', 'መላክ', 'መልአክ', 'ቅዱስ', 'ስብከት', 'ሙስሊም', 
        'ክርስቲያን', 'አላህ', 'ወንጌል', 'ካህን', 'ፓስተር', 'እስልምና', 'ኦርቶዶክስ', 
        'ፕሮቴስታንት', 'መጅሊስ', 'ፆም', 'ጾም', 'በዓል', 'ፈጣሪ', 'ዱአ', 'ሰላት', 
        'ኢማም', 'ሸህ', 'ጳጳስ', 'ኤጲስ ቆጶስ', 'ጥምቀት', 'ፋሲካ', 'ረመዳን', 
        'ቤተ ክርስቲያን', 'ንስሐ', 'ቁርባን', 'ሐዋርያ', 'ነብይ', 'መላእክት', 'መንፈስ ቅዱስ', 
        'መንፈሳዊ', 'ስግደት', 'ምህረት', 'ሐጅ', 'ዑምራ', 'ሰደቃ', 'ዘካ', 'ሰንበት', 
        'ወንጌላዊ', 'መዘምራን', 'መዝሙር', 'የእግዚአብሔር ቃል', 'የእስልምና እምነት', 
        'የክርስትና እምነት', 'መንፈሳዊነት', 'ነፍስ', 'መንፈስ', 'ሰላም', 'ፍቅር', 
        'ይቅርታ', 'ጽድቅ', 'ኃጢአት', 'በደል', 'ክፋት', 'በጎነት', 'ፍርድ', 'ገነት', 
        'ሲኦል', 'መከራ', 'ፈተና', 'በረከት', 'ምስጋና', 'እውነት', 'ሕይወት', 'ሞት', 
        'ካቴድራል', 'ገዳም', 'ሐዲስ ኪዳን', 'ኦሪት', 'ቅዱስ ያሬድ', 'ክብረ በዓል', 
        'ገና', 'ልደት', 'መስቀል', 'ትንሣኤ', 'ዕርገት', 'ዝማሬ', 'ወረብ', 'አቋቋም', 
        'ማህሌት', 'ካቶሊክ', 'ቀሲስ', 'አባ', 'ዲያቆን', 'ፓትርያርክ', 'ጻድቅ', 
        'ማርያም', 'ድንግል', 'ማህበር', 'ታቦት', 'ዕጣን', 'ጸበል', 'ተአምር', 'ተአምራት', 
        'ገድል', 'ስዕለ ማርያም', 'ስዕለ ድንግል', 'መናፍቅ', 'መናፍቃን', 'ኪዳን', 
        'ኪዳነ ምህረት', 'ሙስሊሞች', 'ኢስላም', 'እስላማዊ', 'መሐመድ', 'ረሱል', 
        'ሀዲስ', 'ሐዲስ', 'ሱና', 'ሰላት (ሶላት)', 'ዱዓእ', 'ዒድ', 'ሙፍቲ', 
        'ሼህ', 'ዑለማ', 'ሐላል', 'ሐረም', 'ሐራም', 'ጀነት', 'ጀሀነም', 'ጂብሪል', 
        'ሚካኤል', 'አዝካር', 'ተስቢህ', 'ተራዊህ', 'ጁምዓ', 'ጁማ', 'ውዱእ', 
        'አዛን', 'ኢቃማ', 'መድረሳ', 'ቂርዓት', 'ኪታብ', 'ፈትዋ', 'ቢድዓ', 'ተውሂድ', 
        'ሽርክ', 'የድህነት እምነት'
    ],

    'Health/Medical': [
        # Basic & General Health
        'ሕመም', 'ህመም', 'በሽታ', 'ጤና', 'ጤናማ', 'የጤና', 'የአእምሮ ጤና', 'የጤና ጥበቃ', 
        'ማገገሚያ', 'የጤና ኬላ', 'የጤና ጣቢያ', 'ጤና ጥበቃ ሚኒስቴር', 'የህክምና', 'ሕክምና', 
        'ህክምና', 'ህክምና ቤት', 'ሐኪም ቤት', 'ሕክምና ማዕከል', 'ሆስፒታል', 'ሆስፒታሎች', 
        'ክሊኒክ', 'ፋርማሲ', 'አምቡላንስ', 'አልጋ', 'ድንገተኛ ክፍል', 'የአደጋ ጊዜ ክፍል', 
        'ላብራቶሪ', 'የደም ምርመራ', 'የሽንት ምርመራ', 'ምርመራ', 'ምልክት', 'የህመም ምልክት',

        # Professionals & Roles
        'ሐኪም', 'ሃኪም', 'ዶክተር', 'ዶክተሮች', 'ነርስ', 'ነርሶች', 'ህመምተኛ', 'ሕመምተኛ', 
        'ታካሚ', 'የህክምና ባለሙያ', 'የቀዶ ጥገና ባለሙያ', 'የልብ ሐኪም', 'የዓይን ሐኪም', 
        'የጥርስ ሐኪም', 'የህጻናት ሐኪም', 'የማህፀን ሐኪም', 'የስነ ልቦና ባለሙያ', 'ባለሙያ',

        # Procedures & Equipment
        'ቀዶ ጥገና', 'ቀዶጥገና', 'ክትባት', 'መርፌ', 'ሲሪንጅ', 'ኦክስጅን', 'ኤክስሬይ', 
        'አልትራሳውንድ', 'ስቴቶስኮፕ', 'ቴርሞሜትር', 'ማሰሪያ', 'ፕላስተር', 'የህክምና መሣሪያዎች',

        # Medicines & Treatments
        'መድኃኒት', 'መድሃኒት', 'መድኃኒቶች', 'ጡባዊ', 'ታብሌት', 'ሽሮፕ', 'ቅባት', 
        'ማከሚያ', 'ሕክምና መስጠት',

        # Diseases, Conditions & Infections
        'ካንሰር', 'እጢ', 'ዕጢ', 'ኤድስ', 'ኤችአይቪ', 'ኤች.አይ.ቪ', 'ኮቪድ', 'ባክቴሪያ', 
        'ቫይረስ', 'ተላላፊ', 'ወረርሽኝ', 'ኢንፌክሽን', 'የደም ግፊት', 'ስኳር በሽታ', 
        'ሳል', 'ጉንፋን', 'ወባ', 'ቲዩበርክሎዝ', 'ቲቢ', 'ኮሌራ', 'ታይፎይድ', 'ሄፐታይተስ', 
        'ኤቦላ', 'ኩፍኝ', 'ፉንጎስ', 'አልዛይመር', 'ድብርት', 'ጭንቀት', 'ስብራት', 'ቃጠሎ', 
        'መርዝ', 'የአእምሮ ሕመም', 'ዕብደት',

        # Symptoms & Body context
        'ደም', 'ትኩሳት', 'ራስ ምታት', 'ማስታወክ', 'ማቅለሽለሽ', 'ድካም', 'ማዞር', 
        'እብጠት', 'ደም መፍሰስ', 'የልብ ሕመም', 'የኩላሊት ሕመም', 'የጉበት ሕመም', 
        'የሳንባ ሕመም', 'ሐሞት', 'አንጀት', 'ሆድ',

        # Prevention & Public Health
        'ፀረ ተባይ', 'ንጽህና', 'እጅ መታጠብ', 'የጤና ትምህርት', 'አካላዊ እንቅስቃሴ', 
        'ጤናማ አመጋገብ'
    ],

    'Agriculture': [
        # General & Administrative
        'እርሻ', 'ግብርና', 'አርሶ አደር', 'ገበሬ', 'የገበሬዎች ማህበር', 'የእርሻ ሚኒስቴር', 
        'የግብርና ሚኒስቴር', 'የግብርና ቢሮ', 'የምግብ ዋስትና', 'የመሬት ባለቤትነት', 
        'የግብርና ምርቶች', 'የምግብ ዋስትና ማረጋገጫ', 'የእርሻ ባለሙያ', 'አግሮኖሚ', 
        'የገበሬዎች ማህበር', 'ገጠር ልማት', 'የገጠር ልማት ቢሮ', 'ማስፋፊያ',

        # Land & Soil
        'መሬት', 'እርሻ መሬት', 'ማሳ', 'ማሳዎች', 'አፈር', 'ለስላሳ አፈር', 'የ ለም መሬት', 
        'የሰንበት መሬት', 'የመሬት ይዞታ', 'የመሬት አጠቃቀም', 'የአፈር መሸርሸር', 
        'የአፈር እርጥበት', 'ማዳበሪያ', 'ማዳበሪያዎች', 'ዩሪያ', 'ዲአፒ', 'ኮምፖስት',

        # Seasons & Weather
        'መኸር', 'በጋ', 'በልግ', 'ክረምት', 'ዝናብ', 'የዝናብ እጥረት', 'ድርቅ', 'ጎርፍ', 
        'አየር ንብረት', 'የአየር ሁኔታ', 'የዝናብ ስርጭት', 'መስኖ', 'መስኖ እርሻ', 
        'የከርሰ ምድር ውሃ', 'ግድብ', 'የውሃ ማጠጫ', 'የውሃ ሀብት',

        # Farming Operations & Equipment
        'ማረስ', 'ማረስያ', 'ማረሻ', 'ትራክተር', 'ማጨጃ', 'መውጫ', 'የእርሻ መሣሪያዎች', 
        'አረም', 'አረም ማጥፊያ', 'ፀረ ተባይ', 'ፀረ ተባይ መርጫ', 'ፀረ አረም', 
        'መከር', 'አዝመራ', 'ምርት', 'መሰብሰብ', 'መፍጫ', 'ጎተራ', 'ማከማቻ', 

        # Grains & Cereals (ሰብል እና ጥራጥሬ)
        'ሰብል', 'ሰብሎች', 'ጥራጥሬ', 'ጥራጥሬዎች', 'ስንዴ', 'በቆሎ', 'ጤፍ', 'ገብስ', 
        'ማሽላ', 'ሩዝ', 'ማሾ', 'ባቄላ', 'አተር', 'ምስር', 'ሽምብራ', 'ሉቢያ', 
        'ጐመንዘር', 'ኑግ', 'ተልባ', 'ሰሊጥ', 'ማዕድን ሰብል', 'የስንዴ ምርት', 
        'የቡና ምርት', 'የበቆሎ ምርት', 'የጤፍ ምርት',

        # Cash Crops & Plantations (ጥሬ ዕቃዎችና ተክሎች)
        'ቡና', 'ሻይ', 'ጫት', 'ትንባሆ', 'ጥጥ', 'ጐማ', 'ቅመማቅመም', 'በርበሬ', 
        'ሽንኩርት', 'ነጭ ሽንኩርት', 'ዝንጅብል', 'ቁንዶ በርበሬ', 'ተክሎች', 'ዛፍ', 
        'ችግኝ', 'ችግኝ ማፍያ', 'ደን', 'የደን ልማት', 'ዕፅዋት',

        # Vegetables & Fruits (አትክልትና ፍራፍሬ)
        'አትክልት', 'ፍራፍሬ', 'ድንች', 'ቲማቲም', 'ጐመን', 'ካሮት', 'ፓፓያ', 
        'ማንጎ', 'አቮካዶ', 'ሙዝ', 'ሎሚ', 'ብርቱካን', 'ሐብሐብ', 'አናናስ', 

        # Livestock, Poultry & Animal Husbandry (እንስሳትና እርባታ)
        'ከብት', 'ከብቶች', 'እንስሳት', 'የእንስሳት ሀብት', 'የከብት እርባታ', 'የዶሮ እርባታ', 
        'ንብ እርባታ', 'በግ', 'ፍየል', 'በሬ', 'ላም', 'ጥጃ', 'ግመል', 'ፈረስ', 
        'በቅሎ', 'አህያ', 'ዶሮ', 'ንብ', 'ማር', 'ወተት', 'ቅቤ', 'ጮማ', 'ዕጣ', 
        'የእንስሳት መኖ', 'ሳር', 'ኖላ', 'እረኛ', 'የእንስሳት ሐኪም', 'የእጥበት ቦታ', 
        'የእንስሳት በሽታ'
    ],

  'Business/Economy': [
        # General Trade & Commerce
        'ንግድ', 'ንግድ ሚኒስቴር', 'የንግድ ቢሮ', 'ንግድ ቤት', 'የንግድ ድርጅት', 'የንግድ ማዕከል', 
        'ንግድ ፈቃድ', 'ሽያጭ', 'ግዢ', 'አከፋፋይ', 'ጅምላ', 'ችርቻሮ', 'ገበያ', 'ገበያ ቤት', 
        'የገበያ ማዕከል', 'ሱቅ', 'መደብር', 'ሱፐርማርኬት', 'አምራች', 'ምርት', 'ሸቀጥ', 
        'ሸቀጦች', 'ዕቃ', 'ዕቃዎች', 'ማስታወቂያ', 'ውል', 'ስምምነት', 'ውድድር', 

        # Macroeconomics & Finance
        'ኢኮኖሚ', 'ፋይናንስ', 'በጀት', 'ወጪ', 'ገቢ', 'ሀብት', 'ካፒታል', 'ኢኮኖሚክስ', 
        'የኢኮኖሚ ዕድገት', 'ጠቅላላ የሀገር ውስጥ ምርት', 'ጠህእም', 'ዋጋ', 'የዋጋ ንረት', 
        'የዋጋ ጭማሪ', 'ግሽበት', 'ዋጋ ግሽበት', 'ተመን', 'ግብር', 'ታክስ', 'ጉምሩክ', 
        'ቀረጥ', 'ድጎማ', 'ዕዳ', 'ብድር', 'ወለድ', 'ኪሳራ', 'ትርፍ', 'ትርፍና ኪሳራ', 

        # Banking, Currency & Payments
        'ባንክ', 'ብሔራዊ ባንክ', 'ንግድ ባንክ', 'የባንክ ሂሳብ', 'የገንዘብ ዝውውር', 'ገንዘብ', 
        'ብር', 'ዶላር', 'ዩሮ', 'ፓውንድ', 'የውጭ ምንዛሬ', 'ምንዛሬ', 'ሴቪንግ', 'ቁጠባ', 
        'መለያ ቁጥር', 'አካውንት', 'ቼክ', 'ኤቲኤም', 'ሞባይል ባንኪንግ', 'ዲጂታል ባንኪንግ', 
        'ሐዋላ', 'የገንዘብ ልውውጥ', 'ብድርና ቁጠባ', 'ወለድ አልባ ባንኪንግ', 'ኢንሹራንስ', 
        'ዋስትና', 'አክሲዮን', 'አክሲዮን ማህበር', 'ድርሻ', 'ትርፍ ድርሻ', 'ቦንድ', 

        # Corporate & Industry
        'ኩባንያ', 'ድርጅት', 'ኢንዱስትሪ', 'ፋብሪካ', 'የኢንዱስትሪ ፓርክ', 'የንግድ ማህበር', 
        'የንግድ ምክር ቤት', 'አስተዳዳሪ', 'ስራ አስኪያጅ', 'ሥራ አስፈፃሚ', 'ዳይሬክተር', 
        'ባለሀብት', 'ባለቤት', 'ሰራተኛ', 'ደመወዝ', 'ጉልበት', 'ቀጣሪ', 'ቀጠሮ', 
        'የኢንቨስትመንት ፈቃድ', 'ኢንቨስትመንት', 'ኢንቨስትመንቶች', 'የውጭ ኢንቨስትመንት', 
        'የንግድ ስራ', 'ንግድ እንቅስቃሴ', 'የገበያ ድርሻ', 'የገበያ ጥናት'
    ]
}

# ============================================================================
# CLASSIFICATION LOGIC (Amharic-Only Keyword Matching)
# ============================================================================

def classify_domain(amharic_text):
    am_str = str(amharic_text).lower() if pd.notna(amharic_text) else ""
    scores = defaultdict(int)
    
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in am_str:
                scores[domain] += 1

    if scores:
        return max(scores, key=scores.get)
    return 'General'


def analyze_domains(df):
    print("\n📊 CLASSIFYING DATASET BY DOMAINS (AMHARIC-ONLY KEYWORDS)...")
    
    df['domain'] = [classify_domain(am) for am in df['Amharic']]
    domain_counts = df['domain'].value_counts()
    
    print("\n" + "="*80)
    print("DOMAIN DISTRIBUTION SUMMARY")
    print("="*80)
    print(f"Total Parallel Pairs Analyzed: {len(df):,}\n")
    
    for domain, count in domain_counts.items():
        percentage = (count / len(df)) * 100
        bar = "█" * int(percentage / 2)
        print(f"{domain:18} {count:>9,} rows ({percentage:>5.1f}%) {bar}")
    
    return df, domain_counts


# ============================================================================
# VISUALIZATIONS & REPORTS
# ============================================================================

def plot_domain_distribution(df, domain_counts):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    colors = ['#95a5a6', '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    ax = axes[0]
    ax.pie(domain_counts.values, labels=domain_counts.index, autopct='%1.1f%%',
           colors=colors[:len(domain_counts)], startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax.set_title('Domain Distribution (Pie Chart)', fontsize=14, fontweight='bold', pad=20)
    
    ax = axes[1]
    ax.barh(domain_counts.index, domain_counts.values, color=colors[:len(domain_counts)], edgecolor='black', linewidth=1.2)
    for i, (domain, count) in enumerate(domain_counts.items()):
        ax.text(count, i, f' {count:,} ({count/len(df)*100:.1f}%)', va='center', fontweight='bold', fontsize=10)
    ax.set_xlabel('Number of Sentence Pairs', fontsize=12, fontweight='bold')
    ax.set_title('Domain Distribution (Bar Chart)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(DATA_DIR / '4_domain_distribution.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved Visualization: data/4_domain_distribution.png")
    plt.close()


def plot_domain_characteristics(df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Domain Characteristics Analysis', fontsize=16, fontweight='bold')
    
    domains = df['domain'].unique()
    colors = ['#95a5a6', '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    domain_stats = []
    for domain in sorted(domains):
        domain_df = df[df['domain'] == domain]
        domain_stats.append({
            'domain': domain,
            'am_len': domain_df['Amharic'].fillna("").str.len().mean(),
            'or_len': domain_df['Oromo'].fillna("").str.len().mean(),
            'am_words': domain_df['Amharic'].fillna("").apply(lambda x: len(str(x).split())).mean(),
            'or_words': domain_df['Oromo'].fillna("").apply(lambda x: len(str(x).split())).mean(),
            'count': len(domain_df)
        })
    
    domain_stats_df = pd.DataFrame(domain_stats).set_index('domain')
    x = np.arange(len(domain_stats_df))
    width = 0.35

    ax = axes[0, 0]
    ax.bar(x - width/2, domain_stats_df['am_len'], width, label='Amharic', color='#3498db', edgecolor='black')
    ax.bar(x + width/2, domain_stats_df['or_len'], width, label='Afaan Oromo', color='#e74c3c', edgecolor='black')
    ax.set_ylabel('Avg Characters', fontweight='bold')
    ax.set_title('Average Character Length by Domain', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(domain_stats_df.index, rotation=35, ha='right')
    ax.legend()
    
    ax = axes[0, 1]
    ax.bar(x - width/2, domain_stats_df['am_words'], width, label='Amharic', color='#3498db', edgecolor='black')
    ax.bar(x + width/2, domain_stats_df['or_words'], width, label='Afaan Oromo', color='#e74c3c', edgecolor='black')
    ax.set_ylabel('Avg Words', fontweight='bold')
    ax.set_title('Average Word Count by Domain', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(domain_stats_df.index, rotation=35, ha='right')
    ax.legend()
    
    ax = axes[1, 0]
    ax.bar(domain_stats_df.index, domain_stats_df['count'], color=colors[:len(domain_stats_df)], edgecolor='black')
    ax.set_ylabel('Number of Samples', fontweight='bold')
    ax.set_title('Samples per Domain', fontweight='bold')
    ax.set_xticklabels(domain_stats_df.index, rotation=35, ha='right')
    
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = "DOMAIN SUMMARY METRICS\n" + "="*45 + "\n\n"
    for _, row in domain_stats_df.iterrows():
        summary_text += f"{row.name:18} {int(row['count']):>8,} pairs\n"
        summary_text += f"{'':18} Amharic: {row['am_len']:.0f} chars, {row['am_words']:.1f} words\n"
        summary_text += f"{'':18} Oromo  : {row['or_len']:.0f} chars, {row['or_words']:.1f} words\n\n"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(DATA_DIR / '5_domain_characteristics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved Visualization: data/5_domain_characteristics.png")
    plt.close()


def generate_report(df, domain_counts):
    report_path = DATA_DIR / 'domain_analysis_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("AMHARIC TO AFAAN OROMO DATASET - DOMAIN CLASSIFICATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total Parallel Rows: {len(df):,}\n")
        f.write(f"Generated On: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("DOMAIN BREAKDOWN\n" + "-" * 80 + "\n")
        for domain, count in domain_counts.items():
            percentage = (count / len(df)) * 100
            f.write(f"{domain:18} {count:>9,} pairs ({percentage:>5.1f}%)\n")
            
    print("✓ Saved Report: data/domain_analysis_report.txt")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def find_dataset():
    """Reads raw Excel file am_to_ao.xlsx directly from data/ folder."""
    raw_path = DATA_DIR / "am_to_ao.xlsx"
    if raw_path.exists():
        return raw_path
        
    search_locations = [
        PROJECT_ROOT / "am_to_ao.xlsx",
        SCRIPT_DIR / "am_to_ao.xlsx",
        DATA_DIR / "am_to_ao.csv",
    ]
    
    for file_path in search_locations:
        if file_path.exists():
            return file_path
            
    raise FileNotFoundError("Could not locate 'am_to_ao.xlsx' inside the data/ folder.")


def main():
    print("\n" + "="*80)
    print("STARTING AMHARIC-OROMO DOMAIN CLASSIFICATION (AMHARIC KEYWORDS ONLY)")
    print("="*80)
    
    data_file = find_dataset()
    print(f"\n📂 Loading raw dataset from: {data_file}")
    
    if data_file.suffix == '.xlsx':
        df = pd.read_excel(data_file)
    else:
        df = pd.read_csv(data_file, encoding='utf-8')
        
    print(f"✓ Loaded {len(df):,} parallel sentence pairs.")
    
    # Classify domains using Amharic-only keywords
    df, domain_counts = analyze_domains(df)
    
    # Export output classified dataset directly into data/
    output_dataset_path = DATA_DIR / "classified_am_to_ao.csv"
    df.to_csv(output_dataset_path, index=False, encoding='utf-8')
    print(f"✓ Saved classified dataset to: {output_dataset_path}")
    
    # Visualizations & Report directly into data/
    print("\n📈 Creating visualizations...")
    plot_domain_distribution(df, domain_counts)
    plot_domain_characteristics(df)
    
    print("\n📝 Generating report file...")
    generate_report(df, domain_counts)
    
    print("\n" + "="*80)
    print("✓ ALL DOMAIN CLASSIFICATIONS FINISHED!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()