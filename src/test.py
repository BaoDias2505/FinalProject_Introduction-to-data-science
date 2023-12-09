import re


def _modify_html_template(html_string: str) -> str:
    pattern = r'<tr style="text-align: right;">'
    replacement = r'<tr style="text-align: center;">'
    return re.sub(pattern=pattern, repl=replacement,
                  string=html_string, count=1)


def _add_link(html_string: str) -> str:
    pattern = r'<td>(https://www.youtube.com/watch\?v=[a-zA-Z0-9_-]+)</td>'
    replacement = r'<td><a href="\1">\1</a></td>'
    return re.sub(pattern=pattern, repl=replacement,
                  string=html_string)


print(_add_link("""<tr>
      <th>0</th>
      <td>Luke Barousse</td>
      <td>Become a DATA ANALYST with NO degree?!? The Google Data Analytics Professional Certificate</td>
      <td>https://www.youtube.com/watch?v=fmLPS6FBbac</td>
    </tr>"""))

html_string = """<table border="1" class="dataframe data">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>channelTitle</th>
      <th>title</th>
      <th>URL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Luke Barousse</td>
      <td>Become a DATA ANALYST with NO degree?!? The Google Data Analytics Professional Certificate</td>
      <td>https://www.youtube.com/watch?v=fmLPS6FBbac</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Shashank Kalanithi</td>
      <td>Day in the Life of a Data Analyst - SurveyMonkey Data Transformation</td>
      <td>https://www.youtube.com/watch?v=pKvWD0f18Pc</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Stefanovic</td>
      <td>FASTEST Way to Become a Data Analyst and ACTUALLY Get a Job</td>
      <td>https://www.youtube.com/watch?v=AYWLZ1lES6g</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CS Dojo</td>
      <td>Data Structures &amp; Algorithms #1 - What Are Data Structures?</td>
      <td>https://www.youtube.com/watch?v=bum_19loj9A</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DataCamp</td>
      <td>Turn Your Team into Data Masters</td>
      <td>https://www.youtube.com/watch?v=zer_ZbuaHyE</td>
    </tr>
  </tbody>
</table>"""


# print(_modify_html_template(html_string))
