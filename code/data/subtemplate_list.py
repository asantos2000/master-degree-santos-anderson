# Subtemplates from the book of Witt (2012)
subtemplate_list = [
    {
        "id": "S1",
        "explanation": """

        """,
        "text": """
<operative rule statement subject>::=
{<term>|combination of [<term>, and]|set of <term>}
  {<qualifying clause>|}
        """
    },
    {
        "id": "S2",
        "explanation": """

        """,
        "text": """
<article>::={a |an|the}
        """
    },
    {
        "id": "S3",
        "explanation": """

        """,
        "text": """
<cardinality>::=
{exactly|at least {<positive integer 1> and at most| }}
<positive integer 2>
        """
    },
    {
        "id": "S4",
        "explanation": """
One or more determiners can be used before a noun to provide some information as to which (or how many) instances of the noun's concept are being referred to.
The most commonly used determiners are articles.
Specific determiners are used to limit the noun to referring only to a specific instance or instances.
There is an infinite set of ordinal numbers: 'first', 'second', 'third', etc. Any of these may be used between 'the' and a noun to indicate which member of some sequence is referred to, as in “the first stop”. There are other words or phrases that may be used in a similar way
General determiners limit the number of instances referred to by a noun without being specific as to which instance or instances are referenced. The indefinite articles ('a' and 'an') are general determiners, as are also
1. the cardinal numbers ('one', 'two', 'ten', etc.) and various other words defining how many instances are involved (discussed in Section 5.4.3.1);
2. the quantifiers 'all', 'both', 'every', 'each', 'any', 'no', etc.;
3. 'another', 'other' (but not 'the other', which is a specific determiner);
4. 'either', 'neither'.
There is an infinite set of cardinal numbers: 'one', 'two', 'three', etc. Any of these may be preceded by one of the following:
1. 'exactly' to express a more rigorous statement as to how many instances are involved, as in “each flight booking request must specify exactly one departure date”;
2. 'at least' or 'from', to express the minimum number of instances involved, as in “each flight booking confirmation must specify at least one passenger name”;
3. 'at most' (or 'up to'), to similarly express the maximum number of instances involved.
        """,
        "text": """
<determiner>::=
{<article>|each|that |those|
<cardinality>|at most <positive integer>|
the {{first | last} or only|second or any subsequent|previous|next|same}|
any {other |more}} 
        """
    },
    {
        "id": "S5",
        "explanation": """
A set function is any scalar numeric property of a set of data items.
        """,
        "text": """
<set function>::=
{number |sum| total |maximum|minimum|average|mean|median|
latest | earliest}
        """
    },
    {
        "id": "S6",
        "explanation": """
An inequality operator in a rule statement can be any of the following:
1. 'more than', 'less than', 'later than', 'earlier than', 'no more than', 'no less than', 'no later than', 'no earlier than';
2. 'at least <literal> more than', 'at most <literal> more than', 'at least <literal> later than', 'at most <literal> later than';
3. 'later than <literal> after', 'earlier than <literal> after', 'later than <literal> before', 'earlier than <literal> before'.
        """,
        "text": """
<inequality operator>::=
{{no|} {more|less | later | earlier} than|
at {least |most} <literal> {more| later} than|
{no|} {later | earlier} than <literal> {after |before}}   
        """
    },
    {
        "id": "S7",
        "explanation": """

        """,
        "text": """
<equality operator>::=
{the same as| different from|equal to|unequal to}
        """
    },
    {
        "id": "S8",
        "explanation": """
A transaction signifier is either a term or a reference to a combination of terms, possibly qualified.
        """,
        "text": """
<transaction signifier>::=
{<term>|{record of a| } combination of [<term>, and]}
  {<qualifying clause>| }
        """
    },
    {
        "id": "S9",
        "explanation": """
A verb part may be used wherever 'is' or 'be' may be dropped from a verb phrase.
        """,
        "text": """
<verb part>::=
{<participle>|<adjective>| } <preposition>
        """
    },
    {
        "id": "S10",
        "explanation": """
A predicate is whatever may follow a subject in a clause.
        """,
        "text": """
<predicate>::=
{{<verb phrase>| is {<equality operator>|<inequality operator>}}
  <object>|
<verb phrase> {[<object>, and] | [<object>, or] | }}
        """
    },
    {
        "id": "S11",
        "explanation": """
An object is whatever may follow a verb phrase.
        """,
        "text": """
<object>::=
{{<determiner>|the <set function> of {<determiner>| } | } <term>
  {<qualifying clause>| } |
{<determiner>| } <literal>}
        """
    },
    {
        "id": "S12",
        "explanation": """
An expression is a verbal statement of a calculation.
        """,
        "text": """
<expression>::=
{<object>|
<set function> of {<determiner>| } <term> {<qualifying clause>| } |
<expression> {plus|minus|multiplied by|divided by} <expression>|
{sum|product} of [<expression>, and] |
{square|cube} {root |} of <expression>}
        """
    },
    {
        "id": "S13",
        "explanation": """
Conditional clauses can be used after conditional conjunctions to restrict the scope of a rule statement.
        """,
        "text": """
<conditional clause>::=
{{<determiner> <term> {<qualifying clause>| } |<expression>| it}
{<predicate>| [<predicate> and]| [<predicate> or]} |
[<conditional clause> and] | [<conditional clause> or]}
        """
    },
    {
        "id": "S14",
        "explanation": """
A qualifying clause (also known as a restrictive relative clause) can be used after a term in two ways:
1. Following the subject term of a rule statement, a qualifying clause restricts the scope of that rule statement to a subset of the set of objects signified by that term, rather than the set of all objects signified by that term.
2. Following any other term in a rule statement, a qualifying clause makes any stated constraint more specific than if the qualifying clause were absent.

Thus any of the following can be substituted in place of a <qualifying clause> placeholder:
1. 'that' or 'who' followed by a verb phrase and an optional object (see subtemplate S11): for example, 'that includes an international flight';
2. a verb part (see subtemplate S9) followed by an object: for example, 'specified in a real property transaction';
3. 'other than' followed by either
  a. an object, or
  b. a list of objects, the last two objects in the list separated by 'or' (or a comma and 'or' if adhering to U.S. punctuation standards) and each other pair of objects in the list separated by a comma: for example, 'other than a cash payment or a direct debit payment';
4. a conditional clause (see subtemplate S13) preceded by either
  a. a preposition followed by 'which' or 'whom', or
  b. 'whose': for example, 'for which the payment is made', 'whose name appears on the passport';
5. 'that' or 'who' followed by
  a. a verb phrase,
  b. 'that', 'if', or 'whether', and
  c. a conditional clause: for example, 'who checks whether the passport has expired';
6. a qualifying clause involving multiple criteria:
  a. separated by 'and', with or without 'both', as defined in subtemplates S15 and S16,
  b. separated by 'or', with or without 'either', as defined in subtemplates S17 and S18.

Note that, given the definition of <object> in subtemplate S11, any object in a qualifying clause may itself be qualified by a qualifying clause.
        """,
        "text": """
<qualifying clause>::=
{{that |who} <verb phrase> {<object>| } |
<verb part> <object>|
other than {<object>| [<object>, or]} |
{<preposition> {which|whom}|whose} <conditional clause>|
{that |who} <verb phrase> {that | if |whether} <conditional clause>|
<and-qualifying clause>|
<or-qualifying clause>|
<both-and-qualifying clause>|
<either-or-qualifying clause>}
        """
    },
    {
        "id": "S15",
        "explanation": """

        """,
        "text": """
<and-qualifying clause>::=
{that |who}
{[<verb phrase> {<object>| } and]|
is [<verb part> {<object>| } and]
<verb phrase> [<object> and]}
        """
    },
    {
        "id": "S16",
        "explanation": """

        """,
        "text": """
<both-and-qualifying clause>::=
{that |who}
{both <verb phrase> {<object>| } and <verb phrase> {<object>| } |
{is |are} both <verb part> {<object>| } and <verb part> {<object>| } |
<verb phrase> both <object> and <object>}
        """
    },
    {
        "id": "S17",
        "explanation": """

        """,
        "text": """
<or-qualifying clause>::=
{that |who}
{[<verb phrase> {<object>| } or] |
is [<verb part> {<object>| } or]
<verb phrase> [<object> or]}
        """
    },
    {
        "id": "S18",
        "explanation": """

        """,
        "text": """
<either-or-qualifying clause>::=
{that |who}
{either <verb phrase> {<object>|} or <verb phrase> {<object>| } |
{is |are} either <verb part> {<object>|} or <verb part> {<object>| } |
<verb phrase> either <object> or <object>}
        """
    },
    {
        "id": "S19",
        "explanation": """
The <format definition> placeholder can be replaced by anything defined in the subtemplate
        """,
        "text": """
<format definition>::=
{{exactly|at least |up to} <positive integer 1>|
from <positive integer 2> to <positive integer 3>} <term>
{followed by <format definition>| }
        """
    },
    {
        "id": "S20",
        "explanation": """
The <spatial operator> placeholder can be replaced by anything defined in the subtemplate.
        """,
        "text": """
<spatial operator>::=
{overlap|be within|enclose|span|intersect |meet|be on}
        """
    },
    {
        "id": "S21",
        "explanation": """
Note that the word 'after' may be used as a conjunction (in which case it is shown as 'after') or a preposition (in which case it is shown as 'after'). This is also true of 'before' and 'until'. Other words and phrases operate only as one or the other: for example, 'while' operates only as a conjunction, while 'during', 'within', 'earlier than', and 'later than' operate only as prepositions.
        """,
        "text": """
<time restriction>::=
{at any time| }
{{before| after |during| until |within| {no|} {earlier | later} than} <object>| }
{{before| after |during| until |within|on} <object>|
{before| after |while| until} <conditional clause>}
        """
    },
    {
        "id": "S22",
        "explanation": """
<information access process> defines the form of access that is restricted.
        """,
        "text": """
<information access process>::=
{viewed|created|updated|deleted}
        """
    }
]