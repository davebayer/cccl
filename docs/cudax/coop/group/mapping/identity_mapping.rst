.. _cudax-coop-group-mapping-identity-mapping:

``cudax::coop::identity_mapping``
=================================

.. code:: cuda

    namespace cudax::coop {

    class identity_mapping
    {
    public:
      explicit identity_mapping() = default;

      template <typename Unit, typename ParentGroup, typename PrevMappingResult>
      [[nodiscard]] __device__
      auto map(const Unit&, const ParentGroup&, const PrevMappingResult&) const noexcept;
    };

    } // namespace cudax::coop

Overview
--------

``cudax::coop::identity_mapping`` is a mapping that simply copies the previous mapping result with the exact same values and properties. It's main purpose is for generic programming.

The types is explicitly default constructible.
