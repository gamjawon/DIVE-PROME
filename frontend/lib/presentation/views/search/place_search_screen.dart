import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_svg/svg.dart';
import 'package:frontend/data/models/location_model.dart';
import 'package:frontend/presentation/utils/palette.dart';
import 'package:frontend/presentation/viewmodels/home_viewmodel.dart';
import 'package:frontend/presentation/viewmodels/place_search_viewmodel.dart';

class PlaceSearchScreen extends ConsumerStatefulWidget {
  final String title;
  final String hintText;

  const PlaceSearchScreen({
    super.key,
    required this.title,
    required this.hintText,
  });

  @override
  ConsumerState<PlaceSearchScreen> createState() => _PlaceSearchScreenState();
}

class _PlaceSearchScreenState extends ConsumerState<PlaceSearchScreen> {
  final TextEditingController _searchController = TextEditingController();

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  void _searchPlaces(String query) {
    ref.read(placeSearchViewmodelProvider.notifier).searchPlaces(query);
  }

  void _selectPlace(Location place) {
    Navigator.pop(context, place);
    print(place);
  }

  void _selectCurrentLocation() {
    final locationState = ref.read(locationViewmodelProvider);
    locationState.whenData((location) {
      if (location != null) {
        final currentLocationPlace = Location(
          placeName: '현재 위치',
          addressName: location.addressName,
          roadAddressName: location.roadAddressName,
          longitude: location.longitude,
          latitude: location.latitude,
          categoryName: '현재위치',
        );
        Navigator.pop(context, currentLocationPlace);
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('현재 위치 정보를 가져올 수 없습니다.'),
            backgroundColor: Colors.red,
          ),
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        leading: IconButton(
          icon: Icon(Icons.arrow_back, color: Colors.black),
          onPressed: () => Navigator.pop(context),
        ),
        title: Text(
          widget.title,
          style: TextStyle(
            color: Colors.black,
            fontSize: 18,
            fontFamily: 'Pretendard',
            fontWeight: FontWeight.w600,
          ),
        ),
        centerTitle: true,
      ),
      body: Column(
        children: [
          // 검색 입력 필드
          Container(
            margin: EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: Color(0xFFFD9874), width: 1.5),
            ),
            child: TextField(
              controller: _searchController,
              autofocus: true,
              onChanged: _searchPlaces,
              decoration: InputDecoration(
                hintText: widget.hintText,
                hintStyle: TextStyle(
                  color: Color(0xFF9CA3AF),
                  fontSize: 16,
                  fontFamily: 'Pretendard',
                  fontWeight: FontWeight.w500,
                ),
                border: InputBorder.none,
                contentPadding: EdgeInsets.symmetric(
                  horizontal: 16,
                  vertical: 12,
                ),
                suffixIcon: _searchController.text.isNotEmpty
                    ? IconButton(
                        icon: Icon(Icons.clear, color: Color(0xFF9CA3AF)),
                        onPressed: () {
                          _searchController.clear();
                          _searchPlaces('');
                        },
                      )
                    : Icon(Icons.search, color: Color(0xFF9CA3AF)),
              ),
              style: TextStyle(
                color: Color(0xFF374151),
                fontSize: 16,
                fontFamily: 'Pretendard',
                fontWeight: FontWeight.w500,
              ),
            ),
          ),

          // 현재 위치 버튼
          GestureDetector(
            onTap: _selectCurrentLocation,
            child: Container(
              padding: EdgeInsets.symmetric(vertical: 3, horizontal: 16),
              child: Row(
                children: [
                  SvgPicture.asset(
                    'assets/icons/pin.svg',
                    width: 20,
                    height: 20,
                  ),
                  SizedBox(width: 6),
                  Text(
                    '현위치',
                    style: TextStyle(
                      color: Palette.primaryAccentColor,
                      fontSize: 16,
                      fontFamily: 'Pretendard',
                      fontWeight: FontWeight.w600,
                      height: 1.50,
                      letterSpacing: 0.09,
                    ),
                  ),
                ],
              ),
            ),
          ),

          // 검색 결과 목록
          Expanded(child: _buildSearchResults()),
        ],
      ),
    );
  }

  Widget _buildSearchResults() {
    final placeSearchResponse = ref.watch(placeSearchViewmodelProvider);
    final searchState = placeSearchResponse.when(
      data: (response) => AsyncValue.data(response?.documents ?? []),
      loading: () => const AsyncValue<List<Location>>.loading(),
      error: (error, stackTrace) =>
          AsyncValue<List<Location>>.error(error, stackTrace),
    );

    return searchState.when(
      data: (searchResults) {
        if (_searchController.text.trim().isEmpty) {
          return Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                SvgPicture.asset(
                  'assets/icons/pin.svg',
                  width: 64,
                  height: 64,
                  colorFilter: ColorFilter.mode(
                    Color(0xFFD7D7D7),
                    BlendMode.srcIn,
                  ),
                ),
                SizedBox(height: 16),
                Text(
                  '목적지를 검색해보세요',
                  style: TextStyle(
                    color: Color(0xFFD7D7D7),
                    fontSize: 16,
                    fontFamily: 'Pretendard',
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ],
            ),
          );
        }

        if (searchResults.isEmpty) {
          return Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(Icons.search_off, size: 64, color: Color(0xFFD7D7D7)),
                SizedBox(height: 16),
                Text(
                  '검색 결과가 없습니다',
                  style: TextStyle(
                    color: Color(0xFFD7D7D7),
                    fontSize: 16,
                    fontFamily: 'Pretendard',
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ],
            ),
          );
        }

        return ListView.builder(
          itemCount: searchResults.length,
          padding: EdgeInsets.symmetric(horizontal: 24),
          itemBuilder: (context, index) {
            final place = searchResults[index];
            return _buildPlaceItem(place);
          },
        );
      },
      loading: () => Center(
        child: CircularProgressIndicator(color: Palette.primaryAccentColor),
      ),
      error: (error, _) => Center(
        child: Text(
          '검색 중 오류가 발생했습니다: ${error.toString()}',
          style: TextStyle(
            color: Colors.red,
            fontSize: 16,
            fontFamily: 'Pretendard',
            fontWeight: FontWeight.w500,
          ),
        ),
      ),
    );
  }

  Widget _buildPlaceItem(Location place) {
    return InkWell(
      onTap: () => _selectPlace(place),
      child: Container(
        padding: EdgeInsets.symmetric(vertical: 16, horizontal: 16),
        margin: EdgeInsets.only(bottom: 8),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: Color(0xFFE5E5E5)),
        ),
        child: Row(
          children: [
            Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                color: Color(0xFFF5F5F5),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Icon(
                Icons.location_on,
                color: Palette.primaryAccentColor,
                size: 20,
              ),
            ),
            SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    place.placeName,
                    style: TextStyle(
                      color: Colors.black,
                      fontSize: 16,
                      fontFamily: 'Pretendard',
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  if (place.addressName.isNotEmpty) ...[
                    SizedBox(height: 4),
                    Text(
                      place.addressName,
                      style: TextStyle(
                        color: Color(0xFF666666),
                        fontSize: 14,
                        fontFamily: 'Pretendard',
                        fontWeight: FontWeight.w400,
                      ),
                    ),
                  ],
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
