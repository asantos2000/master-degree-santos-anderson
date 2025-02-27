# Templates from the book of Witt (2012)
# Templates T1 - T5 are not used, they are too general (Witt, 2012, p. 163)
template_list = [
    {
        "id": "T1",
        "explanation": """

        """,
        "text": """
{Each|The} <operative rule statement subject>
must <rule statement predicate>
{{if|unless} <conditional clause>|}.
        """
    },
    {
        "id": "T2",
        "explanation": """

        """,
        "text": """
{A|An|The} <operative rule statement subject>
must not <rule statement predicate>
{{if|unless} <conditional clause>|}.
        """
    },
    {
        "id": "T3",
        "explanation": """

        """,
        "text": """
{Each|The} <operative rule statement subject>
may <rule statement predicate>
only <conditional conjunction> <conditional clause>.
        """
    },
    {
        "id": "T4",
        "explanation": """

        """,
        "text": """
{Each|The} <operative rule statement subject>
must
{({if|unless} <conditional clause>)|}
<rule statement predicate>.
        """
    },
    {
        "id": "T5",
        "explanation": """

        """,
        "text": """
{Each|The} <operative rule statement subject>
may <verb phrase>
only <qualified list>.
        """
    },
    {
        "id": "T6",
        "explanation": """

        """,
        "text": """
{A|An|The|} <definitional rule statement subject>
  {<qualifying clause>|}
<verb phrase> by definition
<definition>.
        """
    },
    {
        "id": "T7",
        "explanation": """

        """,
        "text": """
{A|An} <term 1>
  {of {a|an} <term 2>| }
is by definition
{a|an|the} <term 3>
  <qualifying clause>.
        """
    },
    {
        "id": "T8",
        "explanation": """

        """,
        "text": """
{A|An} <term 1>
  {of {a|an} <term 2>| }
is by definition
[<article> <term 3>, or]
  {of that <term 2>| }.
        """
    },
    {
        "id": "T9",
        "explanation": """

        """,
        "text": """
{<literal 1>|{A|An} <term 1>
  {of {a|an} <term 2>| }}
is by definition
{<literal 2>|
[<literal 3>, or] from a <literal 4> to the following <literal 5>}.
        """
    },
    {
        "id": "T10",
        "explanation": """

        """,
        "text": """
{{A|An} <category attribute term>|
The <category attribute term>
  of {a |an} <entity class term>}
is by definition
{either <literal 1> or <literal 2>|
one of the following: [<literal 3>, or]}.
        """
    },
    {
        "id": "T11",
        "explanation": """

        """,
        "text": """
A transition
  of the <category attribute term> of {a|an} <entity class term>
  from {<literal 1>| [<literal 2>, or]}
  to {<literal 3>| [<literal 4>, or]}
is by definition
impossible.
        """
    },
    {
        "id": "T12",
        "explanation": """

        """,
        "text": """
{A|An} <term 1>
<verb phrase> by definition
{<cardinality>|at most <positive integer>} <term 2>
  {{for |in} {each|the} <term 3>| }.
        """
    },
    {
        "id": "T13",
        "explanation": """

        """,
        "text": """
The <term 1>
  <qualifying clause 1>
is by definition
the same as the <term 2>
  <qualifying clause 2>.
        """
    },
    {
        "id": "T14",
        "explanation": """

        """,
        "text": """
The set of <term 1>
  <qualifying clause 1>
is by definition
the same as the set of <term 1>
  <qualifying clause 2>.
        """
    },
    {
        "id": "T15",
        "explanation": """

        """,
        "text": """
{The| } <attribute term>
  {of {a|an} <entity class term>| }
is by definition
{<inequality operator> <literal 1>
  {and <inequality operator> <literal 2>| } |
  [<literal 3>, or]}.
        """
    },
    {
        "id": "T16",
        "explanation": """

        """,
        "text": """
{The| } <attribute term>
  {of | for} {a|an} <entity class term>
  {<qualifying clause>| }
is by definition calculated as
<expression>.
        """
    },
    {
        "id": "T17",
        "explanation": """

        """,
        "text": """
<literal 1>
is by definition {approximately | } equal to
<literal 2>.
        """
    },
    {
        "id": "T18",
        "explanation": """

        """,
        "text": """
A valid <term>
is by definition composed of
<format definition>.
        """
    },
    {
        "id": "T19",
        "explanation": """

        """,
        "text": """
Each <transaction signifier>
must {specify|contain} <cardinality> <data item term>
  {{in| for} {each|the} <subform term> {(if any)| }
  {<qualifying clause>| } | }
{{if |unless} <conditional clause>| }.
        """
    },
    {
        "id": "T20",
        "explanation": """

        """,
        "text": """
Each <transaction signifier>
must
{({if |unless} <conditional clause>) | }
specify whether {it |{the |each} <term>
  {<qualifying clause>| }}
<verb phrase> [<object>, or].
        """
    },
    {
        "id": "T21",
        "explanation": """

        """,
        "text": """
Each <transaction signifier>
must
{({if |unless} <conditional clause>) | }
specify whether {or not| } {it |{the |each} <term>
  {<qualifying clause>| }}
<verb phrase> {<object>| }.
        """
    },
    {
        "id": "T22",
        "explanation": """

        """,
        "text": """
Each <transaction signifier>
must {specify|contain}
  {{in| for} {each|the} <subform term> {(if any)| }
  {<qualifying clause>| } | }
  {a|an} <data item term 1>, {a|an} <data item term 2>
  {, or|but not} both
{{if |unless} <conditional clause>| }.
        """
    },
    {
        "id": "T23",
        "explanation": """

        """,
        "text": """
Each <transaction signifier>
must
{({if |unless} <conditional clause>) | }
{specify |contain}
  {{in| for} {each|the} <subform term> {(if any)| }
  {<qualifying clause>| } | }
  <cardinality> of the following:
  [<data item term>, or].
        """
    },
    {
        "id": "T24",
        "explanation": """

        """,
        "text": """
{A|An} <transaction signifier>
must not {specify |contain} a <data item term>
  {{in | for} {any|the} <subform term> {(if any)| }
  {<qualifying clause>| } | }
{{if |unless} <conditional clause>| }.
        """
    },
    {
        "id": "T25",
        "explanation": """

        """,
        "text": """
{A|An} <transaction signifier>
must not {specify |contain} more than <positive integer>
  <data item term>
  {{in | for} {any one|the} <subform term> {(if any)| }
  {<qualifying clause>| } | }
{{if |unless} <conditional clause>| }.
        """
    },
    {
        "id": "T26",
        "explanation": """

        """,
        "text": """
The number of <data item term 1>
  {specified|contained}
  {{in| for} {the|each} <subform term> {(if any) | } | }
  in each <transaction signifier>
must be {{no|} {more|less} than|equal to} the <data item term 2>
  {<qualifying clause>| }
{{if |unless} <conditional clause>| }.
        """
    },
    {
        "id": "T27",
        "explanation": """

        """,
        "text": """
{The|Each} <data item term> {(if any)| }
  specified {{in| for} {the|each} <subform term> {(if any)| } | }
  in each <transaction signifier>
must be
  {{other than| } one of the <term> <qualifying clause>| [<literal>, or]}
{{if |unless} <conditional clause>| }.
        """
    },
    {
        "id": "T28",
        "explanation": """

        """,
        "text": """
{The|Each} combination of [<data item term 1>, and] {(if any)| }
  specified {{in| for} {the|each} <subform term> {(if any)| } | }
  in each <transaction signifier>
must be one of the combinations of [<data item term 2>, and]
  {<qualifying clause>| }
{{if |unless} <conditional clause>| }.
        """
    },
    {
        "id": "T29",
        "explanation": """

        """,
        "text": """
{The|Each} <data item term> {(if any)| }
  specified {{in| for} {the|each} <subform term> {(if any)| } | }
  in each <transaction signifier>
must be <inequality operator> <object> {and <inequality operator> <object>| }
{{if |unless} <conditional clause>| }.
        """
    },
    {
        "id": "T30",
        "explanation": """

        """,
        "text": """
{The|Each} <data item term> {(if any)| }
  specified {{in| for} {the|each} <subform term> {(if any)| } | }
  in each <transaction signifier>
must be <equality operator> <object>
{{if |unless} <conditional clause>| }.
        """
    },
    {
        "id": "T31",
        "explanation": """

        """,
        "text": """
The|Each} <data item term 1> {(if any)| }
  <verb part> {the <subform term 1> {(if any)| }
  in|} each <transaction signifier 1>
  {<qualifying clause 1>| }
must be different from the <data item term 1>
  <verb part> {{the |any other} <subform term 1> {(if any)| }
  in| } {that |any other} <transaction signifier 1>
  {<qualifying clause 2>| }
{{if |unless} <conditional clause>| }.
        """
    },
    {
        "id": "T32",
        "explanation": """

        """,
        "text": """
{The|Each} combination of [<data item term 1>, and] {(if any)| }
  <verb part> {the <subform term 1> {(if any)| }
  in| } each <transaction signifier 1>
  {<qualifying clause 1>| }
must be different from the combination of [<data item term 1>, and]
  <verb part> {{the|any other} <subform term 1> {(if any) | }
  in| } {that |any other} <transaction signifier 1>
  {<qualifying clause 2>| }
{{if |unless} <conditional clause>| }.
        """
    },
    {
        "id": "T33",
        "explanation": """

        """,
        "text": """
{The|Each} set of <data item term 1> {(if any)| }
  <verb part> {the <subform term 1> {(if any)| }
  in|} each <transaction signifier 1>
  {<qualifying clause 1>| }
must be different from the set of <data item term 1>
  <verb part> {{the |any other} <subform term 1> {(if any)| }
  in| } {that |any other} <transaction signifier 1>
  {<qualifying clause 2>| }
{{if |unless} <conditional clause>| }.
        """
    },
    {
        "id": "T34",
        "explanation": """

        """,
        "text": """
{The|Each} combination of [<data item term>, and] {(if any)| }
  specified {{in| for} {the|each} <subform term> {(if any)| } | }
  in each <transaction signifier>
  {<qualifying clause>| }
must be such that <conditional clause 1>
{{if |unless} <conditional clause 2>| }.
        """
    },
    {
        "id": "T35",
        "explanation": """

        """,
        "text": """
The <set function> of {the| } <data item term> {(if any)| }
  specified {{in| for} {the|each} <subform term> {(if any)| } | }
  in each <transaction signifier>
  {<qualifying clause>| }
must be {<inequality operator>|<equality operator>} <object>
{{if |unless} <conditional clause>| }.
        """
    },
    {
        "id": "T36",
        "explanation": """

        """,
        "text": """
{The|Each} set of <data item term> {(if any)| }
  specified {{in| for} {the|each} <subform term> {(if any)| } | }
  in each <transaction signifier>
  {<qualifying clause 1>| }
must {be {the same as| different from} |include} the set of <term>
  {<qualifying clause 2>| }
{{if |unless} <conditional clause>| }.
        """
    },
    {
        "id": "T37",
        "explanation": """

        """,
        "text": """
{The|Each} <time period term 1> {(if any)| }
  specified {{in| for} {the|each} <subform term 1> {(if any)| } | }
  in each <transaction signifier 1>
  {<qualifying clause 1>| }
must not overlap the <time period term 1>
  specified {{in| for} {the|each} <subform term 1> {(if any)| } | }
  in any other <transaction signifier 1>
  {<qualifying clause 2>| }
{{if |unless} <conditional clause>| }.
        """
    },
    {
        "id": "T38",
        "explanation": """

        """,
        "text": """
Each <time period term 1>
  within the <time period term 2> {(if any)| }
  specified {{in| for} {the|each} <subform term 1> {(if any)| } | }
  in each <transaction signifier 1>
  {<qualifying clause 1>| }
must be within the <time period term 3>
  specified {{in| for} {the|each} <subform term 2> {(if any)| } | }
  in <cardinality> <transaction signifier 2>
  {<qualifying clause 2>| }
{{if |unless} <conditional clause>| }.
        """
    },
    {
        "id": "T39",
        "explanation": """

        """,
        "text": """
{The|Each} <data item term 1> {(if any)| }
  specified {{in| for} {the|each} <subform term 1> {(if any)| } | }
  in each <transaction signifier 1>
must be different from the <data item term 1>
  specified {{in| for} {the|each} <subform term 1> {(if any)| } | }
  in the latest of the earlier <transaction signifier 1>
{{if |unless} <conditional clause>| }.
        """
    },
    {
        "id": "T40",
        "explanation": """

        """,
        "text": """
{The|Each} combination of [<data item term 1>, and] {(if any)| }
  specified {{in| for} {the|each} <subform term 1> {(if any)| } | }
  in each <transaction signifier 1>
must be different from the combination of [<data item term 1>, and]
  specified {{in| for} {the|each} <subform term 1> {(if any)| } | }
  in the latest of the earlier <transaction signifier 1>
{{if |unless} <conditional clause>| }.
        """
    },
    {
        "id": "T41",
        "explanation": """
Much the same conditions apply to the use of each option or placeholder as in template T27. Note that, if only one day of the week is permitted, use <literal 1>, otherwise list the days of the week separated by commas (with the last two day names in the list separated by ‘or’ or, if adhering to U.S. punctuation conventions, a comma and ‘or’);
The same conditions also apply as in template T27 regarding the fact types that should be present in the fact model.
        """,
        "text": """
{The|Each} <data item term> {(if any)| }
  specified {{in| for} {the|each} <subform term> {(if any)| } | }
  in each <transaction signifier>
must be a {<term>|<literal 1>| [<literal 2>, or]}
{{if |unless} <conditional clause>| }.
        """
    },
    {
        "id": "T42",
        "explanation": """
1. <spatial term 1> is the name of the spatial data item whose values are constrained;
2. if ‘must’ is not followed by ‘not’, ‘Each’ is required if that data item can appear more than once in an instance of the transaction or subform, whereas ‘The’ is required if that data item can appear only once in an instance of the transaction or subform;
3. if ‘must’ is followed by ‘not’, ‘A’ or ‘An’ is required;
4. ‘(if any)’ is required if and only if the data item is optional;
5. the <spatial operator> placeholder can be replaced by anything defined in the subtemplate.
        """,
        "text": """
{The|Each|A|An} <spatial term 1> {(if any)| }
  <qualifying clause 1>
must {not| } <spatial operator> the <spatial term 2>
  <qualifying clause 2>
{{if |unless} <conditional clause>| }.
        """
    },
    {
        "id": "T43",
        "explanation": """
Much the same conditions apply to the use of each option or placeholder as in template T27. In addition
1. ‘represented using’ should be included in the rule statement if the data type is one that admits different representations: for example, the same date can be represented as ‘25/12/2011’, ‘12/25/2011’, or ‘25 December 2011’;
2. <term> must be defined in a standard format definition. The same conditions also apply as in template T27 regarding the fact types that should be present in the fact model.
        """,
        "text": """
The <data item term> {(if any)| }
  specified {{in| for} {the|each} <subform term> {(if any)| } | }
  in each <transaction signifier>
must be {represented using| } a valid <term>
{{if |unless} <conditional clause>| }.
        """
    },
    {
        "id": "T44",
        "explanation": """
<transaction signifier 1> is the term used for the transaction (paper or electronic form or message) or persistent data record which is not allowed to be updated while <transaction signifier 2> is the term used for those objects to which <transaction signifier 1> refers; options for <transaction signifier> are listed in subtemplate S8; There should be a fact type connecting the principal term in each transaction signifier, although the verb phrase in that fact type does not appear in this type of rule statement.
        """,
        "text": """
{A|An} <transaction signifier 1>
must not be transferred
  from one <transaction signifier 2> to another <transaction signifier 2>
{{if |unless} <conditional clause>| }.
        """
    },
    {
        "id": "T45",
        "explanation": """
Much the same conditions apply to the use of each option or placeholder in this template as in template T27. In addition, the term ‘data item’ can be used in place of the <data item term> placeholder.
The same conditions that apply in template T27 regarding the fact types that should be present in the fact model also apply to this template, except when ‘data item’ is used in place of the <data item term> placeholder.
        """,
        "text": """
{The|A|An} <data item term> {(if any)| }
  {{in |for} {any|the} <subform term> {(if any)| } | }
  {in|of} a <transaction signifier>
must not be updated
{{if |unless} <conditional clause>| }.
        """
    },
    {
        "id": "T46",
        "explanation": """
Much the same conditions apply to the use of each option or placeholder as in template T27.
if the rule covers only transitions to a single category, use <literal 1>, otherwise list the categories separated by commas (with the last two categories in the list separated by ‘or’ or, if adhering to U.S. punctuation conventions, a comma and ‘or’).
        """,
        "text": """
The <data item term> {(if any)| }
  {{in |for} {any|the} <subform term> {(if any)| } | }
  {in|of} a <transaction signifier>
may be updated to {<literal 1>| [<literal 2>, or]}
only if <conditional clause>.
        """
    },
    {
        "id": "T47",
        "explanation": """
Much the same conditions apply to the use of each option or placeholder as in template T27.
The same conditions also apply as in template T27 regarding the fact types that should be present in the fact model.
        """,
        "text": """
The <data item term> {(if any)| }
  {{in| for} {any|the} <subform term> {(if any)| } | }
  {in |of} a <transaction signifier>
must not be {increased|decreased}
{{if |unless} <conditional clause>| }.
        """
    },
    {
        "id": "T48",
        "explanation": """
Rule statements in which the subject is the term signifying the process or other activity, such as R385, R386, and R389;
        """,
        "text": """
{The| } <process term> {of | for} {a|an} <object term>
  {<qualifying clause>| }
{must {not| } occur|may occur only}
<time restriction 1> {{and| or} <time restriction 2>| }
{{if |unless} <conditional clause>| }.
        """
    },
    {
        "id": "T49",
        "explanation": """
Rule statements in which the subject is either:
a. the term signifying the object of the process or other activity, such as R387, R388, or R390, or
b. the term signifying the party or device performing the process or other activity.
        """,
        "text": """
{Each|A|An} <term>
  {<qualifying clause 1>| }
{must {not| } <verb phrase 1> {<object 1>| }
  {<qualifying clause 2>| } |
may <verb phrase 2> {<object 2>| }
  {<qualifying clause 3>| } only}
<time restriction 1> {{and| or} <time restriction 2>| }
{{if |unless} <conditional clause>| }.
        """
    },
    {
        "id": "T50",
        "explanation": """
1. <subject term> is the term used for the parties or things that are the focus of the activity;
2. <qualifying clause> is used if not every instance of <subject term> is subject to the rule; the options for <qualifying clause> are listed in subtemplate S14;
3. <verb phrase> is followed by <object> if transitive (e.g., ‘board’, ‘undergo’) but not if intransitive (e.g., ‘be deleted’);
4. <time restriction> is used if the activity is prohibited unless some other activity or event has previously occurred, whereas ‘if <conditional clause>’ is used if the activity is prohibited unless some prerequisite condition exists; the options for <time restriction> are listed in subtemplate S21, while the options for <conditional clause> are listed in subtemplate S13.
        """,
        "text": """
{A|An} <subject term>
  {<qualifying clause>| }
may <verb phrase> {<object>| }
only {<time restriction>| if <conditional clause>}.
        """
    },
    {
        "id": "T51",
        "explanation": """
1. <subject term> is the term used for the parties or things that are the focus of the activity;
2. <qualifying clause> is used if not every instance of <subject term> is subject to the rule; the options for <qualifying clause> are listed in subtemplate S14;
3. <verb phrase> is followed by <object> if transitive (e.g., ‘board’, ‘undergo’) but not if intransitive (e.g., ‘be deleted’);
4. the options for <conditional clause> are listed in subtemplate S13.
        """,
        "text": """
{A|An} <subject term>
  {<qualifying clause>| }
must not <verb phrase> {<object>| }
if <conditional clause>.
        """
    },
    {
        "id": "T52",
        "explanation": """
1. <actor term> is the term signifying the business process or device;
2. <verb phrase> (and <object>, if present) defines the action to be taken in the situation defined by <qualifying clause> or <conditional clause>.
        """,
        "text": """
Each <actor term>
must <verb phrase> {<object>| }
  {<qualifying clause>| }
{{if |unless} <conditional clause>| }.
        """
    },
    {
        "id": "T53",
        "explanation": """
1. <party signifier 1> is a generic term signifying all parties covered by the rule, such as ‘person’, ‘employee’, or ‘organization’;
2. the same term must be substituted in place of each<party signifier 1>;
3. <qualifying clause> is used if the rule governs some proper subset of the parties signified by <party signifier 1>; the options for a <qualifying clause> are listed in subtemplate S14;
4. <predicate 1> signifies the restricted activity or role;
5. ‘the <attribute signifier> of ’ is used if some attribute of a party qualifies that party to perform the activity or play the role;
6. <predicate 2> signifies the value of <attribute signifier>, if used, or some relationship in which a party must participate in order to qualify for the activity or role.
        """,
        "text": """
A <party signifier 1>
  {<qualifying clause>| }
may <predicate 1>
only if {the <attribute signifier> of| } that <party signifier 1>
  <predicate 2>.
        """
    },
    {
        "id": "T54",
        "explanation": """
1. <party signifier 1> is a generic term signifying all parties covered by the rule, such as ‘person’, ‘employee’, or ‘organization’;
2. the same term must be substituted in place of each <party signifier 1>;
3. <qualifying clause 1> defines one of the activities while <qualifying clause 2> defines the other; the options for a <qualifying clause> are listed in subtemplate S14
        """,
        "text": """
The <party signifier 1>
  <qualifying clause 1>
must {not| } be {the same|one of the} <party signifier 1>
  <qualifying clause 2>
{{if |unless} <conditional clause>| }.
        """
    },
    {
        "id": "T55",
        "explanation": """
These templates have the following important options and placeholders:
1. <information signifier> is the term signifying the information item to which access is restricted;
2. <qualifying clause> is used to specify precisely which instances of that information item are subject to restricted access;
3. <information access process> defines the form of access that is restricted; the <information access process> placeholder can be replaced by anything defined in the following subtemplate
        """,
        "text": """
{The|A|An} <information signifier>
  <qualifying clause>
may be <information access process> by
only {<object 1>| [<object 2>, or]}
{{if |unless} <conditional clause>| }.
        """
    },
    {
        "id": "T56",
        "text": """
{The|A|An| } <responsibility signifier>
  {<qualifying clause 1>| }
must <verb phrase> {the |a|an} <party signifier>
  {<qualifying clause 2>| }
{{if |unless} <conditional clause>| }.
        """
    }
]
