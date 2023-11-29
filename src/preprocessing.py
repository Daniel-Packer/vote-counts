import pandas as pd


def preprocess(
    raw_data,
    drop_writein=True,
    drop_undervotes=True,
    drop_overvotes=True,
    drop_contains_undervotes=True,
    drop_contains_overvotes=True,
):
    data = raw_data.copy()
    data = data[~data["writein"]] if drop_writein else data
    data = data.replace("OVER VOTES", "OVERVOTES")
    data = data.replace("UNDER VOTES", "UNDERVOTES")
    data = data[data["candidate"] != "UNDERVOTES"] if drop_undervotes else data
    data = data[data["candidate"] != "OVERVOTES"] if drop_overvotes else data
    

    data["precincts"] = 1
    return data

def get_elections_dict(df, drop_uncontested=True):
  party_dict = (
      df[["candidate", "party_simplified"]]
      .drop_duplicates("candidate")
      .set_index("candidate")["party_simplified"]
      .to_dict()
  )
  election_df = (
      df.groupby(["county_fips", "office", "candidate"])
      .sum(numeric_only=True)["votes"]
      .reset_index()
  )
  elections_dict = {}
  counties = election_df["county_fips"].unique()
  for county in counties:
    county_df = election_df[election_df["county_fips"] == county]
    offices = county_df["office"].unique()
    county_dict = {}
    for office in offices:
      office_df = county_df[county_df["office"] == office].copy()
      office_df["party"] = office_df["candidate"].apply(party_dict.get)
      if not (drop_uncontested and len(office_df) == 1):
        county_dict[office] = office_df[["party", "candidate", "votes"]]
    elections_dict[county] = county_dict
  return elections_dict